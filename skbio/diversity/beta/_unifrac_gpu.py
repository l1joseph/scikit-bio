# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""GPU backend detection for UniFrac (:mod:`skbio.diversity.beta._unifrac_gpu`)."""

import math
from warnings import warn

import numpy as np
from numba import float64

UNWEIGHTED = 0
UNWEIGHTED_UNNORMALIZED = 1
WEIGHTED_NORMALIZED = 2
WEIGHTED_UNNORMALIZED = 3
GENERALIZED = 4

# Block-tiling parameters for the 2D node-chunked kernel (see
# `_make_unifrac_kernel`). A block is (TILE, TILE) threads, i.e. TILE**2
# threads/block. These are backend-specific, not a single global constant,
# because the same kernel source hits different hardware limits on each
# backend at 1024 threads/block (TILE=32):
#
# - AMD/ROCm (``hip``): TILE=32, NODE_CHUNK=32 was chosen empirically on
#   AMD MI300A (gfx942) at n_samples=5000 (~12.5M pairs) against 2 other
#   combinations:
#     TILE=16, NODE_CHUNK=64:  median 1.325s
#     TILE=32, NODE_CHUNK=32:  median 0.952s  <- winner
#     TILE=16, NODE_CHUNK=128: fails to compile -- "local memory (66560)
#         exceeds limit (65536)" (4 shared tiles of TILE*NODE_CHUNK
#         float64 = 16*128*8 = 16KB each = 64KB, right at AMD's 64KB
#         LDS/block limit and pushed over by other locals).
#   (TILE=32, NODE_CHUNK=32) uses 4 * 32*32*8 bytes = 32KB of shared
#   memory per block, well under the 64KB limit, with 1024 threads/block
#   (the max on both AMD and NVIDIA).
# - NVIDIA/CUDA (``cuda``): a 1024-thread block (TILE=32) crashes at
#   kernel launch with ``CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`` on every
#   NVIDIA GPU tested (RTX 2080 Ti/Turing, A10/Ampere) -- this kernel's
#   numba-cuda/NVVM compilation uses more registers/thread than fit
#   NVIDIA's per-SM register budget at 1024 threads/block, even though
#   the identical source is fine on AMD's CU register file. TILE=16
#   (256 threads/block) launches and runs correctly on both NVIDIA cards
#   tested; it is the smallest-change fix (same shared-memory kernel,
#   smaller block) rather than a separate code path.
_TILE_CONFIG = {
    "hip": (32, 32),
    "cuda": (16, 32),
}
_DEFAULT_TILE_CONFIG = (32, 32)


def _get_tile_config(backend):
    """Return the (TILE, NODE_CHUNK) pair tuned for ``backend``.

    Falls back to the AMD-tuned default for an unrecognized backend string
    rather than raising, since the only way to reach kernel code at all is
    via ``get_cuda_module()``, which already restricts ``backend`` to
    ``'cuda'``/``'hip'``.
    """
    return _TILE_CONFIG.get(backend, _DEFAULT_TILE_CONFIG)


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


def _build_block_pair_index(n_samples, tile):
    """Precompute (block_i, block_j) pairs of TILE x TILE sample blocks.

    Unlike `_build_pair_index`, diagonal blocks (``block_i == block_j``) are
    included (``k=0``, not ``k=1``): a diagonal block still contains valid
    ``i < j`` pairs within itself (just not the whole block), so it cannot be
    skipped the way a fully lower-triangle block can.
    """
    num_blocks = (n_samples + tile - 1) // tile
    block_i, block_j = np.triu_indices(num_blocks, k=0)
    return block_i.astype(np.int32), block_j.astype(np.int32)


def _make_unifrac_kernel(cuda, tile, node_chunk):
    """Return the UniFrac pair kernel compiled against a cuda-API module.

    Shared by all 5 UniFrac methods (``UNWEIGHTED``, ``UNWEIGHTED_UNNORMALIZED``,
    ``WEIGHTED_NORMALIZED``, ``WEIGHTED_UNNORMALIZED``, ``GENERALIZED``) so that
    only one kernel is compiled and launched regardless of which method a given
    driver function dispatches.

    ``tile``/``node_chunk`` (see ``_get_tile_config``) are closed over as
    compile-time constants, since the two backends need different values.

    ``cuda.jit`` returns a fresh Dispatcher (and so pays full compilation
    cost, seconds) each time this runs, which would otherwise happen on every
    single ``*_unifrac_gpu`` call. The result is therefore memoized per
    backend module (and tile config) in ``_KERNEL_CACHE``; only the compiled
    kernel object is cached, never any per-call state.

    """
    cache_key = (id(cuda), tile, node_chunk)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached[1]

    TILE = tile
    NODE_CHUNK = node_chunk

    @cuda.jit
    def _unifrac_block_kernel(
        proportions,
        counts,
        sample_totals,
        branch_lengths,
        method,
        alpha,
        variance_adjust,
        block_i,
        block_j,
        n_samples,
        out,
    ):
        blk = cuda.blockIdx.x
        if blk >= block_i.shape[0]:
            return
        bi = block_i[blk]
        bj = block_j[blk]

        # Offset within the i-block / j-block respectively.
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        i = bi * TILE + tx
        j = bj * TILE + ty
        valid = i < n_samples and j < n_samples and i < j

        n_nodes = proportions.shape[1]

        sh_prop_i = cuda.shared.array((TILE, NODE_CHUNK), dtype=float64)
        sh_prop_j = cuda.shared.array((TILE, NODE_CHUNK), dtype=float64)
        sh_cnt_i = cuda.shared.array((TILE, NODE_CHUNK), dtype=float64)
        sh_cnt_j = cuda.shared.array((TILE, NODE_CHUNK), dtype=float64)
        sh_branch = cuda.shared.array(NODE_CHUNK, dtype=float64)

        lin_tid = ty * TILE + tx
        n_threads_blk = TILE * TILE
        tile_elems = TILE * NODE_CHUNK

        numerator = 0.0
        denominator = 0.0

        n_chunks = (n_nodes + NODE_CHUNK - 1) // NODE_CHUNK
        for c in range(n_chunks):
            chunk_start = c * NODE_CHUNK
            chunk_len = n_nodes - chunk_start
            if chunk_len > NODE_CHUNK:
                chunk_len = NODE_CHUNK

            # Cooperative load: every thread in the block (valid or not)
            # helps fill the i-tile and j-tile shared buffers for this
            # node-chunk, strided by the number of threads in the block.
            for e in range(lin_tid, tile_elems, n_threads_blk):
                row = e // NODE_CHUNK
                col = e % NODE_CHUNK
                node = chunk_start + col
                samp_i = bi * TILE + row
                samp_j = bj * TILE + row
                if col < chunk_len and samp_i < n_samples:
                    sh_prop_i[row, col] = proportions[samp_i, node]
                    sh_cnt_i[row, col] = counts[samp_i, node]
                else:
                    sh_prop_i[row, col] = 0.0
                    sh_cnt_i[row, col] = 0.0
                if col < chunk_len and samp_j < n_samples:
                    sh_prop_j[row, col] = proportions[samp_j, node]
                    sh_cnt_j[row, col] = counts[samp_j, node]
                else:
                    sh_prop_j[row, col] = 0.0
                    sh_cnt_j[row, col] = 0.0

            for e in range(lin_tid, NODE_CHUNK, n_threads_blk):
                node = chunk_start + e
                sh_branch[e] = branch_lengths[node] if e < chunk_len else 0.0

            cuda.syncthreads()

            if valid:
                for col in range(chunk_len):
                    p_u = sh_prop_i[tx, col]
                    p_v = sh_prop_j[ty, col]
                    length = sh_branch[col]
                    s = p_u + p_v
                    d = abs(p_u - p_v)
                    if variance_adjust:
                        m = sample_totals[i] + sample_totals[j]
                        mi = sh_cnt_i[tx, col] + sh_cnt_j[ty, col]
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

            # Must finish before the next iteration overwrites the shared
            # tiles just read above.
            cuda.syncthreads()

        if valid:
            idx = i * n_samples - (i * (i + 1)) // 2 + (j - i - 1)
            if method == WEIGHTED_UNNORMALIZED or method == UNWEIGHTED_UNNORMALIZED:
                out[idx] = numerator
            else:
                out[idx] = 0.0 if denominator == 0.0 else numerator / denominator

    # Keep a reference to the module so its id() cannot be reused by another
    # object while this entry lives.
    _KERNEL_CACHE[cache_key] = (cuda, _unifrac_block_kernel)
    return _unifrac_block_kernel


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
    tile, node_chunk = _get_tile_config(detect_gpu_backend())
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
    n_pairs = n_samples * (n_samples - 1) // 2
    block_i, block_j = _build_block_pair_index(n_samples, tile)

    d_proportions = cuda.to_device(proportions)
    d_counts = cuda.to_device(counts_by_node)
    d_sample_totals = cuda.to_device(sample_totals)
    d_branch_lengths = cuda.to_device(branch_lengths.astype(np.float64))
    d_block_i = cuda.to_device(block_i)
    d_block_j = cuda.to_device(block_j)
    d_out = cuda.device_array(n_pairs, dtype=np.float64)

    method = WEIGHTED_NORMALIZED if normalized else WEIGHTED_UNNORMALIZED
    kernel = _make_unifrac_kernel(cuda, tile, node_chunk)
    kernel[block_i.shape[0], (tile, tile)](
        d_proportions,
        d_counts,
        d_sample_totals,
        d_branch_lengths,
        method,
        1.0,
        variance_adjust,
        d_block_i,
        d_block_j,
        n_samples,
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
    tile, node_chunk = _get_tile_config(detect_gpu_backend())
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
    n_pairs = n_samples * (n_samples - 1) // 2
    block_i, block_j = _build_block_pair_index(n_samples, tile)

    d_proportions = cuda.to_device(counts_by_node)
    d_counts = cuda.to_device(counts_by_node)
    d_sample_totals = cuda.to_device(sample_totals)
    d_branch_lengths = cuda.to_device(branch_lengths.astype(np.float64))
    d_block_i = cuda.to_device(block_i)
    d_block_j = cuda.to_device(block_j)
    d_out = cuda.device_array(n_pairs, dtype=np.float64)

    method = UNWEIGHTED if normalized else UNWEIGHTED_UNNORMALIZED
    kernel = _make_unifrac_kernel(cuda, tile, node_chunk)
    kernel[block_i.shape[0], (tile, tile)](
        d_proportions,
        d_counts,
        d_sample_totals,
        d_branch_lengths,
        method,
        1.0,
        variance_adjust,
        d_block_i,
        d_block_j,
        n_samples,
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
    tile, node_chunk = _get_tile_config(detect_gpu_backend())
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
    n_pairs = n_samples * (n_samples - 1) // 2
    block_i, block_j = _build_block_pair_index(n_samples, tile)

    d_proportions = cuda.to_device(proportions)
    d_counts = cuda.to_device(counts_by_node)
    d_sample_totals = cuda.to_device(sample_totals)
    d_branch_lengths = cuda.to_device(branch_lengths.astype(np.float64))
    d_block_i = cuda.to_device(block_i)
    d_block_j = cuda.to_device(block_j)
    d_out = cuda.device_array(n_pairs, dtype=np.float64)

    kernel = _make_unifrac_kernel(cuda, tile, node_chunk)
    kernel[block_i.shape[0], (tile, tile)](
        d_proportions,
        d_counts,
        d_sample_totals,
        d_branch_lengths,
        GENERALIZED,
        alpha,
        variance_adjust,
        d_block_i,
        d_block_j,
        n_samples,
        d_out,
    )
    return d_out.copy_to_host()


# -----------------------------------------------------------------------------
# Two-tier dispatch: fused GPU kernel when usable, array-API fallback otherwise
# -----------------------------------------------------------------------------
#
# The three driver functions above require a real, detected GPU backend
# (``get_cuda_module`` raises ``ImportError`` otherwise) and are left exactly
# as-is: they already run correctly on real NVIDIA/AMD hardware, including
# proactively moving plain NumPy input onto the device for the fused-kernel
# speedup. The wrappers below add the fallback: when no GPU backend is
# detected, or when the fused kernel fails to build/run on the running stack,
# they run the array-API-generic implementation in
# :mod:`skbio.diversity.beta._unifrac_xp` instead, which is correct (if
# slower) on any array-API backend, including plain NumPy on CPU.

# Backends ('cuda'/'hip') whose fused kernel failed to build or run in this
# process. Populated by `_mark_backend_unavailable` so later calls skip
# straight to the array-API path instead of retrying a failing kernel.
# (`skbio.stats.distance._gpu._mark_gpu_unavailable` is the PERMANOVA/Mantel
# precedent for this, but it keys off an array-API array's device/backend;
# UniFrac's GPU dispatch is keyed off `detect_gpu_backend()`'s backend name
# instead -- plain NumPy input has no device of its own -- so a small
# UniFrac-local equivalent is used here rather than reusing that helper.)
_unavailable_backends = set()


def _mark_backend_unavailable(backend):
    """Record that ``backend``'s fused UniFrac kernel cannot run this process.

    Called when the fused kernel raises after a GPU backend was detected
    (for example a numba-hip build that fails to compile on the running ROCm
    stack). Warns once per backend, then routes that backend to the
    array-API fallback from then on.
    """
    if backend not in _unavailable_backends:
        _unavailable_backends.add(backend)
        warn(
            f"The fused UniFrac GPU kernel could not be used for the "
            f"'{backend}' backend on this system; using the array-API "
            "fallback instead.",
            UserWarning,
        )


def _usable_gpu_backend():
    """Return `detect_gpu_backend()`'s result, unless its kernel already failed."""
    backend = detect_gpu_backend()
    if backend in _unavailable_backends:
        return None
    return backend


def weighted_unifrac_gpu_or_xp(
    counts, taxa, tree, normalized, variance_adjust=False, validate=True
):
    """Compute weighted UniFrac, preferring the fused GPU kernel when usable.

    Runs :func:`weighted_unifrac_gpu` when a GPU backend is detected and its
    fused kernel builds/runs successfully; otherwise falls back to
    :func:`skbio.diversity.beta._unifrac_xp.weighted_unifrac_xp`, which is
    correct (if slower) on any array-API backend, including plain NumPy.

    """
    backend = _usable_gpu_backend()
    if backend is not None:
        try:
            return weighted_unifrac_gpu(
                counts, taxa, tree, normalized,
                variance_adjust=variance_adjust, validate=validate,
            )
        except Exception:
            _mark_backend_unavailable(backend)
    from skbio.diversity.beta._unifrac_xp import weighted_unifrac_xp

    return weighted_unifrac_xp(
        counts, taxa, tree, normalized,
        variance_adjust=variance_adjust, validate=validate,
    )


def unweighted_unifrac_gpu_or_xp(
    counts, taxa, tree, normalized, variance_adjust=False, validate=True
):
    """Compute unweighted UniFrac, preferring the fused GPU kernel when usable.

    See :func:`weighted_unifrac_gpu_or_xp`; falls back to
    :func:`skbio.diversity.beta._unifrac_xp.unweighted_unifrac_xp`.

    """
    backend = _usable_gpu_backend()
    if backend is not None:
        try:
            return unweighted_unifrac_gpu(
                counts, taxa, tree, normalized,
                variance_adjust=variance_adjust, validate=validate,
            )
        except Exception:
            _mark_backend_unavailable(backend)
    from skbio.diversity.beta._unifrac_xp import unweighted_unifrac_xp

    return unweighted_unifrac_xp(
        counts, taxa, tree, normalized,
        variance_adjust=variance_adjust, validate=validate,
    )


def generalized_unifrac_gpu_or_xp(
    counts, taxa, tree, alpha, variance_adjust=False, validate=True
):
    """Compute generalized UniFrac, preferring the fused GPU kernel when usable.

    See :func:`weighted_unifrac_gpu_or_xp`; falls back to
    :func:`skbio.diversity.beta._unifrac_xp.generalized_unifrac_xp`.

    """
    backend = _usable_gpu_backend()
    if backend is not None:
        try:
            return generalized_unifrac_gpu(
                counts, taxa, tree, alpha,
                variance_adjust=variance_adjust, validate=validate,
            )
        except Exception:
            _mark_backend_unavailable(backend)
    from skbio.diversity.beta._unifrac_xp import generalized_unifrac_xp

    return generalized_unifrac_xp(
        counts, taxa, tree, alpha,
        variance_adjust=variance_adjust, validate=validate,
    )
