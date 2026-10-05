# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main

import numpy as np

from skbio import DistanceMatrix
from skbio.diversity.beta._unifrac_gpu import (
    _KERNEL_CACHE,
    _build_pair_index,
    _get_tile_config,
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

# unweighted_unifrac's normalized=False and variance_adjust=True, and
# weighted_unifrac's variance_adjust=True, are GPU-only in this release (the
# corresponding CPU/numba kernel support was reverted; see CHANGELOG.md), so
# those combinations are checked against the qiime-191-tt SSU-fixture
# distance matrices directly instead of against the CPU numba kernel.
# Measured max abs deviation, GPU vs ssu-ascii-fixture, is far looser than
# the GPU-vs-CPU-numba figure above; see test_unifrac.py's
# SSU_FIXTURE_TOLERANCE for the same figure used on the CPU side.
GPU_FIXTURE_TOLERANCE = 1.5e-6


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


class GetTileConfigTests(TestCase):
    """`_get_tile_config` picks the right (TILE, NODE_CHUNK) per backend.

    AMD's 1024-thread block (32, 32) crashes at launch on NVIDIA with
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (register budget), so the two
    backends must not resolve to the same config.
    """

    def test_hip_keeps_the_amd_tuned_config(self):
        self.assertEqual(_get_tile_config("hip"), (32, 32))

    def test_cuda_uses_a_smaller_block(self):
        tile, _ = _get_tile_config("cuda")
        self.assertLess(tile, 32)

    def test_hip_and_cuda_configs_differ(self):
        self.assertNotEqual(_get_tile_config("hip"), _get_tile_config("cuda"))

    def test_unrecognized_backend_falls_back_to_amd_default(self):
        self.assertEqual(_get_tile_config("bogus"), (32, 32))


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
        cache_key = (id(fake), 32, 32)
        self.addCleanup(_KERNEL_CACHE.pop, cache_key, None)
        first = _make_unifrac_kernel(fake, 32, 32)
        second = _make_unifrac_kernel(fake, 32, 32)
        self.assertIs(first, second)

    def test_distinct_backends_get_distinct_kernels(self):
        fake1 = self._fake_cuda_module()
        fake2 = self._fake_cuda_module()
        self.addCleanup(_KERNEL_CACHE.pop, (id(fake1), 32, 32), None)
        self.addCleanup(_KERNEL_CACHE.pop, (id(fake2), 32, 32), None)
        self.assertIsNot(
            _make_unifrac_kernel(fake1, 32, 32), _make_unifrac_kernel(fake2, 32, 32)
        )

    def test_distinct_tile_configs_get_distinct_kernels(self):
        fake = self._fake_cuda_module()
        self.addCleanup(_KERNEL_CACHE.pop, (id(fake), 32, 32), None)
        self.addCleanup(_KERNEL_CACHE.pop, (id(fake), 16, 32), None)
        self.assertIsNot(
            _make_unifrac_kernel(fake, 32, 32), _make_unifrac_kernel(fake, 16, 32)
        )


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

    def test_weighted_unifrac_gpu_matches_fixture_variance_adjusted_unnormalized(self):
        # weighted_unifrac's variance_adjust is GPU-only in this release (the
        # CPU/numba kernel's variance_adjust support was reverted; see
        # CHANGELOG.md), so this checks the GPU result against the
        # qiime-191-tt SSU fixture directly rather than against the CPU
        # kernel.
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        gpu = weighted_unifrac_gpu(
            table, taxa, tree,
            normalized=False, variance_adjust=True, validate=True,
        )
        obs = DistanceMatrix(gpu, sample_ids)
        expected = self._load_dm_fixture('weighted_unifrac_vaw_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=GPU_FIXTURE_TOLERANCE)

    def test_weighted_unifrac_gpu_matches_fixture_variance_adjusted_normalized(self):
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        gpu = weighted_unifrac_gpu(
            table, taxa, tree,
            normalized=True, variance_adjust=True, validate=True,
        )
        obs = DistanceMatrix(gpu, sample_ids)
        expected = self._load_dm_fixture('weighted_normalized_unifrac_vaw_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=GPU_FIXTURE_TOLERANCE)


class UnweightedUnifracGpuTests(QiimeTinyTestMixin, TestCase):
    # unweighted_unifrac's normalized and variance_adjust kwargs are GPU-only
    # in this release (the CPU/numba kernel was reverted to its pre-PR,
    # always-normalized, no-variance_adjust behavior; see CHANGELOG.md), so
    # all four combinations here are checked against the qiime-191-tt SSU
    # fixture distance matrices directly rather than against the CPU kernel.

    def setUp(self):
        if detect_gpu_backend() is None:
            self.skipTest("no GPU backend available")

    def test_unweighted_unifrac_gpu_matches_fixture_unnormalized(self):
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        gpu = unweighted_unifrac_gpu(
            table, taxa, tree,
            normalized=False, variance_adjust=False, validate=True,
        )
        obs = DistanceMatrix(gpu, sample_ids)
        expected = self._load_dm_fixture('unweighted_unnormalized_unifrac_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=GPU_FIXTURE_TOLERANCE)

    def test_unweighted_unifrac_gpu_matches_fixture_normalized(self):
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        gpu = unweighted_unifrac_gpu(
            table, taxa, tree,
            normalized=True, variance_adjust=False, validate=True,
        )
        obs = DistanceMatrix(gpu, sample_ids)
        expected = self._load_dm_fixture('unweighted_unifrac_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=GPU_FIXTURE_TOLERANCE)

    def test_unweighted_unifrac_gpu_matches_fixture_variance_adjusted_unnormalized(
        self,
    ):
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        gpu = unweighted_unifrac_gpu(
            table, taxa, tree,
            normalized=False, variance_adjust=True, validate=True,
        )
        obs = DistanceMatrix(gpu, sample_ids)
        expected = self._load_dm_fixture(
            'unweighted_unnormalized_unifrac_vaw_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=GPU_FIXTURE_TOLERANCE)

    def test_unweighted_unifrac_gpu_matches_fixture_variance_adjusted_normalized(
        self,
    ):
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        gpu = unweighted_unifrac_gpu(
            table, taxa, tree,
            normalized=True, variance_adjust=True, validate=True,
        )
        obs = DistanceMatrix(gpu, sample_ids)
        expected = self._load_dm_fixture('unweighted_unifrac_vaw_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=GPU_FIXTURE_TOLERANCE)


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
