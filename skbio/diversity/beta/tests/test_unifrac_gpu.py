# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main

import numpy as np

from skbio.diversity.beta._unifrac_gpu import (
    _KERNEL_CACHE,
    _build_pair_index,
    _make_unifrac_kernel,
    detect_gpu_backend,
    generalized_unifrac_gpu,
    get_cuda_module,
    unweighted_unifrac_gpu,
    weighted_unifrac_gpu,
)
from skbio.diversity.beta.tests._fixtures import QiimeTinyTestMixin

# Measured max abs deviation, GPU vs CPU-numba, across all 12
# method/normalized/variance_adjust combinations on the qiime-191-tt table:
# 3.331e-16 (ordinary float64 noise), confirmed on real NVIDIA and AMD
# hardware -- see
# docs/superpowers/specs/2026-10-02-ssu-unifrac-numba-phase1-design.md.
# 1e-10 gives a huge (~3e5x) safety margin.
GPU_CPU_TOLERANCE = 1e-10


class GpuBackendTests(TestCase):

    def test_detect_gpu_backend_returns_known_value(self):
        self.assertIn(detect_gpu_backend(), ("cuda", "hip", None))

    def test_get_cuda_module_raises_cleanly_when_no_gpu(self):
        backend = detect_gpu_backend()
        if backend is not None:
            self.skipTest("a GPU backend is available in this environment")
        with self.assertRaises(ImportError):
            get_cuda_module()


class BuildPairIndexTests(TestCase):
    """GPU-independent coverage for `_build_pair_index` (no GPU required)."""

    def test_matches_scipy_pdist_condensed_order(self):
        from scipy.spatial.distance import squareform

        n = 5
        pair_i, pair_j = _build_pair_index(n)
        n_pairs = n * (n - 1) // 2
        self.assertEqual(pair_i.shape, (n_pairs,))
        self.assertEqual(pair_j.shape, (n_pairs,))

        # Reconstruct which (i, j) scipy's condensed order expects at each
        # index via squareform, and check _build_pair_index agrees exactly.
        square = np.arange(n * n).reshape(n, n)
        condensed = squareform(square, checks=False)
        for idx in range(n_pairs):
            i, j = pair_i[idx], pair_j[idx]
            self.assertLess(i, j)
            self.assertEqual(condensed[idx], square[i, j])

    def test_dtype_and_trivial_cases(self):
        for n in (0, 1, 2, 3):
            pair_i, pair_j = _build_pair_index(n)
            expected_len = n * (n - 1) // 2
            self.assertEqual(len(pair_i), expected_len)
            self.assertEqual(len(pair_j), expected_len)
            self.assertEqual(pair_i.dtype, np.int32)
            self.assertEqual(pair_j.dtype, np.int32)


class MakeUnifracKernelTests(TestCase):
    """Kernel compilation is memoized per backend (no GPU required)."""

    def _fake_cuda_module(self):
        class FakeCuda:
            """Stands in for numba.cuda; `jit` just returns the function."""

            @staticmethod
            def jit(func):
                return func

            @staticmethod
            def grid(ndim):  # pragma: no cover - never called
                return 0

        return FakeCuda()

    def test_kernel_is_compiled_once_per_backend(self):
        fake = self._fake_cuda_module()
        self.addCleanup(_KERNEL_CACHE.pop, id(fake), None)
        first = _make_unifrac_kernel(fake)
        second = _make_unifrac_kernel(fake)
        self.assertIs(first, second)

    def test_distinct_backends_get_distinct_kernels(self):
        fake1 = self._fake_cuda_module()
        fake2 = self._fake_cuda_module()
        self.addCleanup(_KERNEL_CACHE.pop, id(fake1), None)
        self.addCleanup(_KERNEL_CACHE.pop, id(fake2), None)
        self.assertIsNot(_make_unifrac_kernel(fake1), _make_unifrac_kernel(fake2))


class WeightedUnifracGpuTests(QiimeTinyTestMixin, TestCase):

    def setUp(self):
        if detect_gpu_backend() is None:
            self.skipTest("no GPU backend available")

    def test_weighted_unifrac_gpu_matches_cpu_unnormalized(self):
        from skbio.diversity.beta._unifrac import _weighted_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = weighted_unifrac_gpu(
            table, taxa, tree,
            normalized=False, variance_adjust=False, validate=True,
        )
        cpu = _weighted_unifrac_pdist_numba(
            table, taxa, tree, normalized=False, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)

    def test_weighted_unifrac_gpu_matches_cpu_normalized(self):
        from skbio.diversity.beta._unifrac import _weighted_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = weighted_unifrac_gpu(
            table, taxa, tree,
            normalized=True, variance_adjust=False, validate=True,
        )
        cpu = _weighted_unifrac_pdist_numba(
            table, taxa, tree, normalized=True, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)

    def test_weighted_unifrac_gpu_matches_cpu_variance_adjusted_unnormalized(self):
        from skbio.diversity.beta._unifrac import _weighted_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = weighted_unifrac_gpu(
            table, taxa, tree,
            normalized=False, variance_adjust=True, validate=True,
        )
        cpu = _weighted_unifrac_pdist_numba(
            table, taxa, tree,
            normalized=False, variance_adjust=True, validate=True,
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)

    def test_weighted_unifrac_gpu_matches_cpu_variance_adjusted_normalized(self):
        from skbio.diversity.beta._unifrac import _weighted_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = weighted_unifrac_gpu(
            table, taxa, tree,
            normalized=True, variance_adjust=True, validate=True,
        )
        cpu = _weighted_unifrac_pdist_numba(
            table, taxa, tree,
            normalized=True, variance_adjust=True, validate=True,
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)


class UnweightedUnifracGpuTests(QiimeTinyTestMixin, TestCase):

    def setUp(self):
        if detect_gpu_backend() is None:
            self.skipTest("no GPU backend available")

    def test_unweighted_unifrac_gpu_matches_cpu_unnormalized(self):
        from skbio.diversity.beta._unifrac import _unweighted_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = unweighted_unifrac_gpu(
            table, taxa, tree,
            normalized=False, variance_adjust=False, validate=True,
        )
        cpu = _unweighted_unifrac_pdist_numba(
            table, taxa, tree, normalized=False, variance_adjust=False, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)

    def test_unweighted_unifrac_gpu_matches_cpu_normalized(self):
        from skbio.diversity.beta._unifrac import _unweighted_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = unweighted_unifrac_gpu(
            table, taxa, tree,
            normalized=True, variance_adjust=False, validate=True,
        )
        cpu = _unweighted_unifrac_pdist_numba(
            table, taxa, tree, normalized=True, variance_adjust=False, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)

    def test_unweighted_unifrac_gpu_matches_cpu_variance_adjusted_unnormalized(self):
        from skbio.diversity.beta._unifrac import _unweighted_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = unweighted_unifrac_gpu(
            table, taxa, tree,
            normalized=False, variance_adjust=True, validate=True,
        )
        cpu = _unweighted_unifrac_pdist_numba(
            table, taxa, tree, normalized=False, variance_adjust=True, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)

    def test_unweighted_unifrac_gpu_matches_cpu_variance_adjusted_normalized(self):
        from skbio.diversity.beta._unifrac import _unweighted_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = unweighted_unifrac_gpu(
            table, taxa, tree,
            normalized=True, variance_adjust=True, validate=True,
        )
        cpu = _unweighted_unifrac_pdist_numba(
            table, taxa, tree, normalized=True, variance_adjust=True, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)


class GeneralizedUnifracGpuTests(QiimeTinyTestMixin, TestCase):

    def setUp(self):
        if detect_gpu_backend() is None:
            self.skipTest("no GPU backend available")

    def test_generalized_unifrac_gpu_matches_cpu_alpha_half(self):
        from skbio.diversity.beta._unifrac import _generalized_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = generalized_unifrac_gpu(
            table, taxa, tree, alpha=0.5, variance_adjust=False, validate=True,
        )
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=0.5, variance_adjust=False, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)

    def test_generalized_unifrac_gpu_matches_cpu_alpha_one(self):
        from skbio.diversity.beta._unifrac import _generalized_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = generalized_unifrac_gpu(
            table, taxa, tree, alpha=1.0, variance_adjust=False, validate=True,
        )
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=1.0, variance_adjust=False, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)

    def test_generalized_unifrac_gpu_matches_cpu_variance_adjusted(self):
        from skbio.diversity.beta._unifrac import _generalized_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = generalized_unifrac_gpu(
            table, taxa, tree, alpha=0.5, variance_adjust=True, validate=True,
        )
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=0.5, variance_adjust=True, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)

    def test_generalized_unifrac_gpu_matches_cpu_variance_adjusted_alpha_one(self):
        from skbio.diversity.beta._unifrac import _generalized_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = generalized_unifrac_gpu(
            table, taxa, tree, alpha=1.0, variance_adjust=True, validate=True,
        )
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=1.0, variance_adjust=True, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=GPU_CPU_TOLERANCE)


if __name__ == "__main__":
    main()
