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
    _build_pair_index,
    detect_gpu_backend,
    generalized_unifrac_gpu,
    get_cuda_module,
    unweighted_unifrac_gpu,
    weighted_unifrac_gpu,
)
from skbio.diversity.beta.tests._fixtures import QiimeTinyTestMixin


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
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)

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
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)


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
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)

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
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)

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
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)

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
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)


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
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)

    def test_generalized_unifrac_gpu_matches_cpu_alpha_one(self):
        from skbio.diversity.beta._unifrac import _generalized_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = generalized_unifrac_gpu(
            table, taxa, tree, alpha=1.0, variance_adjust=False, validate=True,
        )
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=1.0, variance_adjust=False, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)

    def test_generalized_unifrac_gpu_matches_cpu_variance_adjusted(self):
        from skbio.diversity.beta._unifrac import _generalized_unifrac_pdist_numba
        table, taxa, tree, _ = self._load_qiime_191_tt()
        gpu = generalized_unifrac_gpu(
            table, taxa, tree, alpha=0.5, variance_adjust=True, validate=True,
        )
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=0.5, variance_adjust=True, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)


if __name__ == "__main__":
    main()
