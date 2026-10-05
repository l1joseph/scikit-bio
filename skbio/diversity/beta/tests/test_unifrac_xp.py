# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

from skbio import DistanceMatrix
from skbio.diversity.beta._unifrac import (
    _weighted_unifrac_pdist_numba,
    _unweighted_unifrac_pdist_numba,
    _generalized_unifrac_pdist_numba,
)
from skbio.diversity.beta._unifrac_gpu import (
    generalized_unifrac_gpu_or_xp,
    unweighted_unifrac_gpu_or_xp,
    weighted_unifrac_gpu_or_xp,
)
from skbio.diversity.beta._unifrac_xp import (
    generalized_unifrac_xp,
    unweighted_unifrac_xp,
    weighted_unifrac_xp,
)
from skbio.diversity.beta.tests._fixtures import QiimeTinyTestMixin

# Measured max abs deviation, array-API path vs CPU-numba kernel, for the
# method/normalized combinations both implement (no variance_adjust) on the
# qiime-191-tt table: ~3.3e-16 (ordinary float64 noise). 1e-10 matches the
# tolerance used for the equivalent GPU-vs-CPU-numba comparison in
# test_unifrac_gpu.py.
XP_CPU_TOLERANCE = 1e-10

# variance_adjust and unweighted_unifrac's normalized=False have no CPU/numba
# counterpart (GPU-only in this release), so those combinations are checked
# against the qiime-191-tt SSU fixture distance matrices directly instead,
# using the same tolerance as test_unifrac_gpu.py's GPU_FIXTURE_TOLERANCE.
XP_FIXTURE_TOLERANCE = 1.5e-6


class WeightedUnifracXpTests(QiimeTinyTestMixin, TestCase):

    def test_weighted_unifrac_xp_matches_cpu_unnormalized(self):
        table, taxa, tree, _ = self._load_qiime_191_tt()
        xp_res = weighted_unifrac_xp(
            table, taxa, tree, normalized=False, validate=True)
        cpu = _weighted_unifrac_pdist_numba(
            table, taxa, tree, normalized=False, validate=True)
        np.testing.assert_allclose(xp_res, cpu, rtol=0, atol=XP_CPU_TOLERANCE)

    def test_weighted_unifrac_xp_matches_cpu_normalized(self):
        table, taxa, tree, _ = self._load_qiime_191_tt()
        xp_res = weighted_unifrac_xp(
            table, taxa, tree, normalized=True, validate=True)
        cpu = _weighted_unifrac_pdist_numba(
            table, taxa, tree, normalized=True, validate=True)
        np.testing.assert_allclose(xp_res, cpu, rtol=0, atol=XP_CPU_TOLERANCE)

    def test_weighted_unifrac_xp_matches_fixture_variance_adjusted_unnormalized(self):
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        xp_res = weighted_unifrac_xp(
            table, taxa, tree, normalized=False, variance_adjust=True,
            validate=True,
        )
        obs = DistanceMatrix(xp_res, sample_ids)
        expected = self._load_dm_fixture('weighted_unifrac_vaw_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=XP_FIXTURE_TOLERANCE)

    def test_weighted_unifrac_xp_matches_fixture_variance_adjusted_normalized(self):
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        xp_res = weighted_unifrac_xp(
            table, taxa, tree, normalized=True, variance_adjust=True,
            validate=True,
        )
        obs = DistanceMatrix(xp_res, sample_ids)
        expected = self._load_dm_fixture('weighted_normalized_unifrac_vaw_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=XP_FIXTURE_TOLERANCE)


class UnweightedUnifracXpTests(QiimeTinyTestMixin, TestCase):

    def test_unweighted_unifrac_xp_matches_cpu_normalized(self):
        table, taxa, tree, _ = self._load_qiime_191_tt()
        xp_res = unweighted_unifrac_xp(
            table, taxa, tree, normalized=True, validate=True)
        cpu = _unweighted_unifrac_pdist_numba(table, taxa, tree, validate=True)
        np.testing.assert_allclose(xp_res, cpu, rtol=0, atol=XP_CPU_TOLERANCE)

    def test_unweighted_unifrac_xp_matches_fixture_unnormalized(self):
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        xp_res = unweighted_unifrac_xp(
            table, taxa, tree, normalized=False, validate=True)
        obs = DistanceMatrix(xp_res, sample_ids)
        expected = self._load_dm_fixture('unweighted_unnormalized_unifrac_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=XP_FIXTURE_TOLERANCE)

    def test_unweighted_unifrac_xp_matches_fixture_variance_adjusted_unnormalized(
        self,
    ):
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        xp_res = unweighted_unifrac_xp(
            table, taxa, tree, normalized=False, variance_adjust=True,
            validate=True,
        )
        obs = DistanceMatrix(xp_res, sample_ids)
        expected = self._load_dm_fixture(
            'unweighted_unnormalized_unifrac_vaw_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=XP_FIXTURE_TOLERANCE)

    def test_unweighted_unifrac_xp_matches_fixture_variance_adjusted_normalized(self):
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        xp_res = unweighted_unifrac_xp(
            table, taxa, tree, normalized=True, variance_adjust=True,
            validate=True,
        )
        obs = DistanceMatrix(xp_res, sample_ids)
        expected = self._load_dm_fixture('unweighted_unifrac_vaw_dm.txt')
        np.testing.assert_allclose(
            obs.filter(sample_ids).data, expected.filter(sample_ids).data,
            rtol=0, atol=XP_FIXTURE_TOLERANCE)


class GeneralizedUnifracXpTests(QiimeTinyTestMixin, TestCase):

    def test_generalized_unifrac_xp_matches_cpu_alpha_half(self):
        table, taxa, tree, _ = self._load_qiime_191_tt()
        xp_res = generalized_unifrac_xp(
            table, taxa, tree, alpha=0.5, variance_adjust=False, validate=True)
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=0.5, variance_adjust=False, validate=True)
        np.testing.assert_allclose(xp_res, cpu, rtol=0, atol=XP_CPU_TOLERANCE)

    def test_generalized_unifrac_xp_matches_cpu_alpha_one(self):
        table, taxa, tree, _ = self._load_qiime_191_tt()
        xp_res = generalized_unifrac_xp(
            table, taxa, tree, alpha=1.0, variance_adjust=False, validate=True)
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=1.0, variance_adjust=False, validate=True)
        np.testing.assert_allclose(xp_res, cpu, rtol=0, atol=XP_CPU_TOLERANCE)

    def test_generalized_unifrac_xp_matches_cpu_variance_adjusted(self):
        table, taxa, tree, _ = self._load_qiime_191_tt()
        xp_res = generalized_unifrac_xp(
            table, taxa, tree, alpha=0.5, variance_adjust=True, validate=True)
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=0.5, variance_adjust=True, validate=True)
        np.testing.assert_allclose(xp_res, cpu, rtol=0, atol=XP_CPU_TOLERANCE)

    def test_generalized_unifrac_xp_matches_cpu_variance_adjusted_alpha_one(self):
        table, taxa, tree, _ = self._load_qiime_191_tt()
        xp_res = generalized_unifrac_xp(
            table, taxa, tree, alpha=1.0, variance_adjust=True, validate=True)
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=1.0, variance_adjust=True, validate=True)
        np.testing.assert_allclose(xp_res, cpu, rtol=0, atol=XP_CPU_TOLERANCE)


class GpuOrXpDispatchTests(QiimeTinyTestMixin, TestCase):
    """Unit coverage for the `*_gpu_or_xp` dispatch wrappers themselves.

    Forces the no-usable-backend branch by monkeypatching
    `detect_gpu_backend` to return None, independent of what hardware this
    machine actually has, and checks the result against the CPU reference.
    Public-API-level coverage of the same fallback (through
    unweighted_unifrac/weighted_unifrac/generalized_unifrac and
    beta_diversity) lives in test_unifrac.py.
    """

    def test_weighted_unifrac_gpu_or_xp_falls_back_without_gpu(self):
        table, taxa, tree, _ = self._load_qiime_191_tt()
        with patch(
            "skbio.diversity.beta._unifrac_gpu.detect_gpu_backend",
            return_value=None,
        ):
            obs = weighted_unifrac_gpu_or_xp(
                table, taxa, tree, normalized=True, validate=True)
        cpu = _weighted_unifrac_pdist_numba(
            table, taxa, tree, normalized=True, validate=True)
        np.testing.assert_allclose(obs, cpu, rtol=0, atol=XP_CPU_TOLERANCE)

    def test_unweighted_unifrac_gpu_or_xp_falls_back_without_gpu(self):
        table, taxa, tree, _ = self._load_qiime_191_tt()
        with patch(
            "skbio.diversity.beta._unifrac_gpu.detect_gpu_backend",
            return_value=None,
        ):
            obs = unweighted_unifrac_gpu_or_xp(
                table, taxa, tree, normalized=True, validate=True)
        cpu = _unweighted_unifrac_pdist_numba(table, taxa, tree, validate=True)
        np.testing.assert_allclose(obs, cpu, rtol=0, atol=XP_CPU_TOLERANCE)

    def test_generalized_unifrac_gpu_or_xp_falls_back_without_gpu(self):
        # generalized_unifrac has no CPU implementation reachable from the
        # public API, so this does not raise when no GPU backend is usable
        # -- it is the regression this whole fallback exists to fix.
        table, taxa, tree, _ = self._load_qiime_191_tt()
        with patch(
            "skbio.diversity.beta._unifrac_gpu.detect_gpu_backend",
            return_value=None,
        ):
            obs = generalized_unifrac_gpu_or_xp(
                table, taxa, tree, alpha=0.5, validate=True)
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=0.5, validate=True)
        np.testing.assert_allclose(obs, cpu, rtol=0, atol=XP_CPU_TOLERANCE)


if __name__ == "__main__":
    main()
