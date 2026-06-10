# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Tests for OptSpace matrix completion."""

import unittest

import numpy as np
import numpy.testing as npt

from skbio.stats.ordination import optspace
from skbio.util import numba_code


def _low_rank_masked(n=30, m=20, rank=2, frac_missing=0.3, seed=0):
    """Build a known low-rank matrix with a fraction of entries masked (NaN)."""
    rng = np.random.default_rng(seed)
    truth = rng.standard_normal((n, rank)) @ rng.standard_normal((rank, m))
    obs = truth.copy()
    obs[rng.random(obs.shape) < frac_missing] = np.nan
    return truth, obs


class TestOptSpace(unittest.TestCase):
    def test_shapes(self):
        _, obs = _low_rank_masked()
        n, m = obs.shape
        r = 2
        U, s, V, dist = optspace(obs, n_components=r, max_iterations=50)
        self.assertEqual(U.shape, (n, r))
        self.assertEqual(s.shape, (r, r))
        self.assertEqual(V.shape, (m, r))
        self.assertEqual(dist.shape, (n, n))

    def test_completes_low_rank(self):
        # A rank-2 matrix completed at rank 2 should recover observed entries
        # well; the per-entry RMSE over observed entries should be small.
        truth, obs = _low_rank_masked(rank=2)
        mask = ~np.isnan(obs)
        U, s, V, _ = optspace(obs, n_components=2, max_iterations=200, tol=1e-8)
        completed = U @ s @ V.T
        rmse = np.sqrt(np.mean((completed[mask] - truth[mask]) ** 2))
        self.assertLess(rmse, 1e-2)

    def test_singular_values_descending(self):
        _, obs = _low_rank_masked()
        _, s, _, _ = optspace(obs, n_components=3, max_iterations=50)
        diag = np.diag(s)
        npt.assert_array_equal(diag, np.sort(diag)[::-1])

    def test_distance_symmetric_zero_diag(self):
        _, obs = _low_rank_masked()
        _, _, _, dist = optspace(obs, n_components=2, max_iterations=50)
        npt.assert_allclose(dist, dist.T, atol=1e-12)
        npt.assert_allclose(np.diag(dist), 0.0, atol=1e-12)

    def test_reproducible(self):
        _, obs = _low_rank_masked()
        out1 = optspace(obs, n_components=2, max_iterations=50)
        out2 = optspace(obs, n_components=2, max_iterations=50)
        # ARPACK (svds) init is not guaranteed bit-for-bit across calls, so the
        # outputs agree to machine precision rather than exactly.
        for a, b in zip(out1, out2):
            npt.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_default_engine_runs(self):
        # Explicitly exercise the default (cython/NumPy) path.
        _, obs = _low_rank_masked()
        U, s, V, dist = optspace(
            obs, n_components=2, max_iterations=50, engine="cython"
        )
        self.assertFalse(np.isnan(U).any())
        self.assertFalse(np.isnan(dist).any())

    def test_n_components_not_int(self):
        _, obs = _low_rank_masked()
        with self.assertRaisesRegex(ValueError, "integer"):
            optspace(obs, n_components=2.5, max_iterations=5)

    def test_n_components_too_large(self):
        _, obs = _low_rank_masked(n=10, m=8)
        with self.assertRaisesRegex(ValueError, "at most 1 minus"):
            optspace(obs, n_components=8, max_iterations=5)

    def test_n_components_nonpositive(self):
        _, obs = _low_rank_masked()
        with self.assertRaisesRegex(ValueError, "positive"):
            optspace(obs, n_components=0, max_iterations=5)

    def test_not_two_dimensional(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            optspace(np.arange(10.0), n_components=2, max_iterations=5)

    def test_bad_engine(self):
        _, obs = _low_rank_masked()
        with self.assertRaisesRegex(ValueError, "not supported"):
            optspace(obs, n_components=2, max_iterations=5, engine="julia")

    @numba_code
    def test_numba_matches_default(self):
        _, obs = _low_rank_masked(n=40, m=25, rank=3)
        out_np = optspace(
            obs, n_components=3, max_iterations=50, tol=1e-8, engine="cython"
        )
        out_nb = optspace(
            obs, n_components=3, max_iterations=50, tol=1e-8, engine="numba"
        )
        for a, b in zip(out_np, out_nb):
            npt.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
