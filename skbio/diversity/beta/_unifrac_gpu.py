# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""GPU backend detection for UniFrac (:mod:`skbio.diversity.beta._unifrac_gpu`)."""

_backend_cache = None


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
