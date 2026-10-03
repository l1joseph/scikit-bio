# GPU-agnostic UniFrac Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `generalized_unifrac` (GUniFrac) and `variance_adjust` support to scikit-bio's UniFrac metrics, with a GPU-agnostic numba kernel (one kernel source, running on NVIDIA via `numba-cuda` and AMD via `numba.hip`'s `pose_as_cuda()`) as the primary implementation and the existing CPU numba kernel extended as a secondary fallback, covering all 5 SSU UniFrac methods.

**Architecture:** Formulas are read directly from SSU's C++ source (`unifrac-binaries` v1.7) and verified against the real `ssu` binary's output via checked-in fixtures, never re-derived from assumption. CPU kernels extend the existing row+mirror-parallelized pattern in `skbio/diversity/beta/_unifrac.py`. GPU kernels live in a new `skbio/diversity/beta/_unifrac_gpu.py`, written once against the `numba.cuda` API, with a thin runtime backend-detection layer choosing `numba-cuda` or `numba.hip` — never two separate kernel implementations.

**Tech Stack:** Python, Numba (`njit`/`prange` for CPU, `numba.cuda` API for GPU), `numba-cuda` (new optional dependency), `numba.hip`/`hip-python` (new optional dependency), pytest, the real `ssu` binary (dev-only, via the `unifrac-bench` conda env) for fixture generation, `kubectl`/NRP for NVIDIA verification, native `srun` on Cosmos for AMD verification.

**Spec:** `docs/superpowers/specs/2026-10-02-ssu-unifrac-numba-phase1-design.md`

## Global Constraints

- Pure Python/numba only. No new C++ build dependency for scikit-bio. `unifrac-binaries`/`ssu` are dev-only reference tools, never a runtime or test-suite dependency.
- `fastmath=False` on both CPU and GPU kernels by default — exactness matters, SSU was chosen over DartUniFrac partly because it's exact.
- `generalized_unifrac` and `variance_adjust=True` are numba-only (CPU or GPU), no cython fallback: raise `ImportError` when numba isn't installed, matching the existing pattern in commit `94b40ae5`.
- One kernel body for GPU, written against `numba.cuda`, backend-selected at runtime — never maintain separate CUDA-specific and HIP-specific kernel source.
- Output stays condensed (vector form) throughout; construct `DistanceMatrix(condensed, ids)` directly, no `squareform()` call.
- Alpha for `generalized_unifrac` must be in `[0, 1]`, default `1.0`.
- All new formulas are summed directly over every tree node (tips + internal) using plain `branch_lengths`, reusing the arrays `_setup_multiple_unifrac` already produces — the existing `node_to_root_distances` tip-only optimization does not generalize to these nonlinear formulas and must not be used for them.

---

## Task 1: Generate SSU reference fixtures

**Files:**
- Create (dev-only, not shipped): `/cosmos/vast/scratch/l1joseph/unifrac-bench/scripts/gen_fixtures.sh`
- Create (checked in): `skbio/diversity/beta/tests/data/qiime-191-tt/unweighted_unnormalized_unifrac_dm.txt`
- Create (checked in): `skbio/diversity/beta/tests/data/qiime-191-tt/generalized_unifrac_alpha0.5_dm.txt`
- Create (checked in): `skbio/diversity/beta/tests/data/qiime-191-tt/generalized_unifrac_alpha1.0_dm.txt`
- Create (checked in): `skbio/diversity/beta/tests/data/qiime-191-tt/unweighted_unifrac_vaw_dm.txt`
- Create (checked in): `skbio/diversity/beta/tests/data/qiime-191-tt/unweighted_unnormalized_unifrac_vaw_dm.txt`
- Create (checked in): `skbio/diversity/beta/tests/data/qiime-191-tt/weighted_unifrac_vaw_dm.txt`
- Create (checked in): `skbio/diversity/beta/tests/data/qiime-191-tt/weighted_normalized_unifrac_vaw_dm.txt`
- Create (checked in): `skbio/diversity/beta/tests/data/qiime-191-tt/generalized_unifrac_alpha1.0_vaw_dm.txt`

**Interfaces:**
- Produces: 8 new plain-text distance matrix fixture files, same tab-separated format as the existing `unweighted_unifrac_dm.txt` (header row of sample IDs, one row per sample), used by every later task's correctness tests.

- [ ] **Step 1: Convert the existing tiny-test table to BIOM**

Run inside the `unifrac-bench` conda env (has `biom-format` installed as a dependency of the `unifrac` package):

```bash
cd /cosmos/vast/scratch/l1joseph/unifrac-bench
mkdir -p scripts fixtures
conda run -n unifrac-bench biom convert \
  -i /cosmos/nfs/home/l1joseph/scikit-bio/skbio/diversity/beta/tests/data/qiime-191-tt/otu-table.tsv \
  -o fixtures/qiime-191-tt.biom \
  --table-type="OTU table" --to-hdf5
```

- [ ] **Step 2: Write the fixture-generation script**

```bash
cat > scripts/gen_fixtures.sh <<'EOF'
#!/bin/bash
set -euo pipefail
BENCH=/cosmos/vast/scratch/l1joseph/unifrac-bench
TREE=/cosmos/nfs/home/l1joseph/scikit-bio/skbio/diversity/beta/tests/data/qiime-191-tt/tree.nwk
TABLE=$BENCH/fixtures/qiime-191-tt.biom
OUT=$BENCH/fixtures

run() {
  local method=$1 outfile=$2; shift 2
  conda run -n unifrac-bench ssu -i "$TABLE" -t "$TREE" -m "$method" -o "$OUT/$outfile" -r ascii "$@"
}

run unweighted_unnormalized unweighted_unnormalized_unifrac_dm.txt
run generalized generalized_unifrac_alpha0.5_dm.txt -a 0.5
run generalized generalized_unifrac_alpha1.0_dm.txt -a 1.0
run unweighted unweighted_unifrac_vaw_dm.txt --vaw
run unweighted_unnormalized unweighted_unnormalized_unifrac_vaw_dm.txt --vaw
run weighted_unnormalized weighted_unifrac_vaw_dm.txt --vaw
run weighted_normalized weighted_normalized_unifrac_vaw_dm.txt --vaw
run generalized generalized_unifrac_alpha1.0_vaw_dm.txt -a 1.0 --vaw
EOF
chmod +x scripts/gen_fixtures.sh
./scripts/gen_fixtures.sh
```

- [ ] **Step 3: Verify output format matches the existing fixtures**

```bash
head -2 /cosmos/vast/scratch/l1joseph/unifrac-bench/fixtures/unweighted_unnormalized_unifrac_dm.txt
head -2 /cosmos/nfs/home/l1joseph/scikit-bio/skbio/diversity/beta/tests/data/qiime-191-tt/unweighted_unifrac_dm.txt
```
Expected: same tab-separated structure (header row of 8 sample IDs matching `f2 f1 f3 f4 p2 p1 t1 t2`, 8 data rows).

- [ ] **Step 4: Copy fixtures into the repo and commit**

```bash
cp /cosmos/vast/scratch/l1joseph/unifrac-bench/fixtures/*_dm.txt \
   /cosmos/nfs/home/l1joseph/scikit-bio/skbio/diversity/beta/tests/data/qiime-191-tt/
cd /cosmos/nfs/home/l1joseph/scikit-bio
git add skbio/diversity/beta/tests/data/qiime-191-tt/*_dm.txt
git commit -m "test: add ssu-generated reference fixtures for generalized/vaw unifrac"
```

---

## Task 2: CPU — `unweighted_unifrac` gains `normalized` and `variance_adjust`

**Files:**
- Modify: `skbio/diversity/beta/_unifrac.py:37-160` (`unweighted_unifrac` public function, `_unweighted_unifrac` private function)
- Modify: `skbio/diversity/beta/_unifrac.py:528-647` (numba kernels `_unweighted_unifrac_row_nb`, `_unweighted_unifrac_pdist_nb`, `_unweighted_unifrac_pdist_numba`)
- Test: `skbio/diversity/beta/tests/test_unifrac.py`

**Interfaces:**
- Produces: `unweighted_unifrac(u_counts, v_counts, taxa, tree, normalized=True, variance_adjust=False, validate=True) -> float`. When `variance_adjust=True`, requires numba (raises `ImportError` otherwise).
- Consumes: `vectorize_counts_and_tree` (existing, unchanged), `NUMBA_AVAILABLE` (existing module flag).

- [ ] **Step 1: Write the failing tests**

```python
# in skbio/diversity/beta/tests/test_unifrac.py, inside UnifracTests

def test_unweighted_unifrac_unnormalized_matches_ssu_fixture(self):
    table, taxa, tree, sample_ids = self._load_qiime_191_tt()
    expected = self._load_dm_fixture('unweighted_unnormalized_unifrac_dm.txt')
    for i, j in [(0, 1), (2, 5), (3, 7)]:
        obs = unweighted_unifrac(
            table[i], table[j], taxa, tree, normalized=False
        )
        self.assertAlmostEqual(obs, expected[sample_ids[i], sample_ids[j]], places=5)

def test_unweighted_unifrac_variance_adjust_matches_ssu_fixture(self):
    table, taxa, tree, sample_ids = self._load_qiime_191_tt()
    expected = self._load_dm_fixture('unweighted_unifrac_vaw_dm.txt')
    for i, j in [(0, 1), (2, 5), (3, 7)]:
        obs = unweighted_unifrac(
            table[i], table[j], taxa, tree, variance_adjust=True
        )
        self.assertAlmostEqual(obs, expected[sample_ids[i], sample_ids[j]], places=5)

@skipIf(NUMBA_AVAILABLE, "numba is installed")
def test_unweighted_unifrac_variance_adjust_requires_numba(self):
    with self.assertRaises(ImportError):
        unweighted_unifrac(
            [1, 0, 1], [0, 1, 1], ['a', 'b', 'c'], self.t1,
            variance_adjust=True,
        )

def test_unweighted_unifrac_both_empty_unnormalized_is_zero(self):
    obs = unweighted_unifrac(
        [0, 0, 0], [0, 0, 0], ['a', 'b', 'c'], self.t1, normalized=False
    )
    self.assertEqual(obs, 0.0)

def test_unweighted_unifrac_both_empty_variance_adjust_is_zero(self):
    obs = unweighted_unifrac(
        [0, 0, 0], [0, 0, 0], ['a', 'b', 'c'], self.t1, variance_adjust=True
    )
    self.assertEqual(obs, 0.0)
```

Create a new shared mixin file so both the CPU tests here and the GPU tests in Task 7-9 use the identical loading code, not duplicated or inconsistently redefined:

```python
# skbio/diversity/beta/tests/_fixtures.py
"""Shared QIIME 1.9.1 tiny-test fixture loading for unifrac tests."""

import pandas as pd

from skbio import TreeNode, DistanceMatrix
from skbio.util._testing import get_data_path


class QiimeTinyTestMixin:
    """Mixin providing access to the qiime-191-tt fixture directory."""

    def _load_qiime_191_tt(self):
        base = get_data_path('qiime-191-tt', subfolder='data')
        df = pd.read_csv(f'{base}/otu-table.tsv', sep='\t', skiprows=1, index_col=0)
        taxa = df.index.astype(str).tolist()
        sample_ids = df.columns.tolist()
        table = df.T.values
        tree = TreeNode.read(f'{base}/tree.nwk')
        return table, taxa, tree, sample_ids

    def _load_dm_fixture(self, filename):
        base = get_data_path('qiime-191-tt', subfolder='data')
        return DistanceMatrix.read(f'{base}/{filename}')
```

(Grep `skbio/util/_testing.py` for `get_data_path`'s exact signature before trusting the `subfolder=` keyword verbatim — confirm it against an existing caller elsewhere in the test suite, since it is not currently imported in `test_unifrac.py`.)

Change `class UnifracTests(TestCase):` (line 27) to `class UnifracTests(QiimeTinyTestMixin, TestCase):` and add `from skbio.diversity.beta.tests._fixtures import QiimeTinyTestMixin` to the existing import block.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /cosmos/nfs/home/l1joseph/scikit-bio
python -m pytest skbio/diversity/beta/tests/test_unifrac.py -k "unnormalized_matches_ssu or variance_adjust_matches_ssu or variance_adjust_requires_numba" -v
```
Expected: FAIL — `unweighted_unifrac() got an unexpected keyword argument 'normalized'` (and similarly for `variance_adjust`).

- [ ] **Step 3: Extend the private per-pair function**

Replace `_unweighted_unifrac` (currently lines 347-378) with a version that supports both toggles for the single-pair (non-numba) path, used when `validate=True`'s slow path or no-numba fallback is taken:

```python
def _unweighted_unifrac(
    u_node_counts, v_node_counts, branch_lengths, normalized=True,
    variance_adjust=False, u_total_count=None, v_total_count=None,
):
    unique_nodes = np.logical_xor(u_node_counts, v_node_counts)
    observed_nodes = np.logical_or(u_node_counts, v_node_counts)
    if variance_adjust:
        # m is the scalar total tip-count sum per sample (same value used for
        # every node, matching run_VawUnweightedTask_T's sample_total_counts[k]);
        # mi varies per node (embedded_counts[offset+k]).
        m = u_total_count + v_total_count
        mi = u_node_counts.astype(np.float64) + v_node_counts
        with np.errstate(invalid="ignore"):
            vaw = np.sqrt(mi * (m - mi))
        weight = np.where(vaw > 0, branch_lengths / np.where(vaw > 0, vaw, 1.0), 0.0)
        unique_branch_length = (weight * unique_nodes).sum()
        observed_branch_length = (weight * observed_nodes).sum()
    else:
        unique_branch_length = (branch_lengths * unique_nodes).sum()
        observed_branch_length = (branch_lengths * observed_nodes).sum()
    if not normalized:
        return unique_branch_length
    if observed_branch_length == 0.0:
        return 0.0
    return unique_branch_length / observed_branch_length
```

`u_total_count`/`v_total_count` are the scalar per-sample tip-count totals (`_setup_pairwise_unifrac`'s existing `total_counts` return value, currently discarded by `unweighted_unifrac` via `_, _` — Step 4 below captures them instead), not the per-node count arrays; `mi` is computed per node from `u_node_counts`/`v_node_counts` directly inside this function.

- [ ] **Step 4: Update the public function signature**

```python
@params_aliased([("taxa", "otu_ids", "0.6.0", True)])
def unweighted_unifrac(
    u_counts, v_counts, taxa, tree, normalized=True, variance_adjust=False,
    validate=True,
):
    if variance_adjust and not NUMBA_AVAILABLE:
        raise ImportError(
            "variance_adjust=True for unweighted_unifrac requires numba."
        )
    (
        u_node_counts, v_node_counts, u_total_count, v_total_count, tree_index,
    ) = _setup_pairwise_unifrac(
        u_counts, v_counts, taxa, tree, validate, normalized=False, unweighted=True
    )
    return _unweighted_unifrac(
        u_node_counts, v_node_counts, tree_index["length"],
        normalized=normalized, variance_adjust=variance_adjust,
        u_total_count=u_total_count, v_total_count=v_total_count,
    )
```

- [ ] **Step 5: Extend the numba pdist kernel**

Modify `_unweighted_unifrac_row_nb` and `_unweighted_unifrac_pdist_nb` (lines 535-628) to take `normalized: bool` and `variance_adjust: bool` flags and a `sample_totals` array (reuse the pattern already present in `_weighted_unifrac_row_nb`/`_weighted_unifrac_pdist_nb` for how `sample_totals` is threaded through):

```python
@njit(inline="always")
def _unweighted_unifrac_row_nb(
    row, n_samples, n_nodes, counts_by_node, branch_lengths, sample_totals,
    normalized, variance_adjust, out,
):
    base = _condensed_row_base(row, n_samples)
    for j in range(row + 1, n_samples):
        unique = 0.0
        observed = 0.0
        for k in range(n_nodes):
            u_count = counts_by_node[row, k]
            v_count = counts_by_node[j, k]
            u_present = u_count > 0
            v_present = v_count > 0
            if not (u_present or v_present):
                continue
            bl = branch_lengths[k]
            if variance_adjust:
                m = sample_totals[row] + sample_totals[j]
                mi = u_count + v_count
                vaw = np.sqrt(mi * (m - mi))
                if vaw <= 0.0:
                    continue
                bl = bl / vaw
            observed += bl
            if u_present != v_present:
                unique += bl
        idx = base + j
        if not normalized:
            out[idx] = unique
        elif observed == 0.0:
            out[idx] = 0.0
        else:
            out[idx] = unique / observed
```

Thread `normalized`, `variance_adjust`, `sample_totals` through `_unweighted_unifrac_pdist_nb`'s signature and both call sites (the `i` row and `mirror_i` row), the same way `_weighted_unifrac_pdist_nb` already threads `normalized`. Update `_unweighted_unifrac_pdist_numba` to compute `sample_totals` the same way `_weighted_unifrac_pdist_numba` does (`counts_by_node[:, tip_indices].sum(axis=1, dtype=np.float64)`, reusing `_get_tip_indices`) and pass the two new flags through.

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac.py -k "unweighted_unifrac" -v
```
Expected: PASS, all `unweighted_unifrac` tests including the pre-existing ones (confirming no regression).

- [ ] **Step 7: Commit**

```bash
git add skbio/diversity/beta/_unifrac.py skbio/diversity/beta/tests/test_unifrac.py
git commit -m "feat: add normalized and variance_adjust to unweighted_unifrac"
```

---

## Task 3: CPU — `weighted_unifrac` gains `variance_adjust`

**Files:**
- Modify: `skbio/diversity/beta/_unifrac.py:164-327` (`weighted_unifrac` public function)
- Modify: `skbio/diversity/beta/_unifrac.py:702-824` (`_weighted_unifrac_row_nb`, `_weighted_unifrac_pdist_nb`, `_weighted_unifrac_pdist_numba`)
- Test: `skbio/diversity/beta/tests/test_unifrac.py`

**Interfaces:**
- Produces: `weighted_unifrac(u_counts, v_counts, taxa, tree, normalized=False, variance_adjust=False, validate=True) -> float`.
- Consumes: same `_setup_pairwise_unifrac`/`_setup_multiple_unifrac` as Task 2, `sample_totals` pattern from Task 2.

- [ ] **Step 1: Write the failing tests**

```python
def test_weighted_unifrac_variance_adjust_matches_ssu_fixture_unnormalized(self):
    table, taxa, tree, sample_ids = self._load_qiime_191_tt()
    expected = self._load_dm_fixture('weighted_unifrac_vaw_dm.txt')
    for i, j in [(0, 1), (2, 5), (3, 7)]:
        obs = weighted_unifrac(table[i], table[j], taxa, tree, variance_adjust=True)
        self.assertAlmostEqual(obs, expected[sample_ids[i], sample_ids[j]], places=5)

def test_weighted_unifrac_variance_adjust_matches_ssu_fixture_normalized(self):
    table, taxa, tree, sample_ids = self._load_qiime_191_tt()
    expected = self._load_dm_fixture('weighted_normalized_unifrac_vaw_dm.txt')
    for i, j in [(0, 1), (2, 5), (3, 7)]:
        obs = weighted_unifrac(
            table[i], table[j], taxa, tree, normalized=True, variance_adjust=True
        )
        self.assertAlmostEqual(obs, expected[sample_ids[i], sample_ids[j]], places=5)

def test_weighted_unifrac_both_empty_variance_adjust_normalized_is_zero(self):
    obs = weighted_unifrac(
        [0, 0, 0], [0, 0, 0], ['a', 'b', 'c'], self.t1,
        normalized=True, variance_adjust=True,
    )
    self.assertEqual(obs, 0.0)

def test_weighted_unifrac_both_empty_variance_adjust_unnormalized_is_zero(self):
    obs = weighted_unifrac(
        [0, 0, 0], [0, 0, 0], ['a', 'b', 'c'], self.t1, variance_adjust=True
    )
    self.assertEqual(obs, 0.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac.py -k "weighted_unifrac_variance_adjust" -v
```
Expected: FAIL — unexpected keyword argument `variance_adjust`.

- [ ] **Step 3: Add a VAW branch to the single-pair path**

Add a new private function (VAW's per-node sum is over *all* nodes with plain `branch_lengths`, per the Global Constraints note — do not reuse `node_to_root_distances` here):

```python
def _weighted_unifrac_vaw(
    u_node_counts, v_node_counts, u_total_count, v_total_count, branch_lengths,
    normalized,
):
    m = u_total_count + v_total_count
    mi = u_node_counts.astype(np.float64) + v_node_counts
    with np.errstate(invalid="ignore"):
        vaw = np.sqrt(mi * (m - mi))
    mask = vaw > 0
    safe_vaw = np.where(mask, vaw, 1.0)
    if u_total_count > 0:
        up = u_node_counts / u_total_count
    else:
        up = u_node_counts.astype(np.float64)
    if v_total_count > 0:
        vp = v_node_counts / v_total_count
    else:
        vp = v_node_counts.astype(np.float64)
    numerator = np.where(mask, branch_lengths * np.abs(up - vp) / safe_vaw, 0.0).sum()
    if not normalized:
        return numerator
    denominator = np.where(mask, branch_lengths * (up + vp) / safe_vaw, 0.0).sum()
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
```

Wire it into `weighted_unifrac` (replacing the body from Task 1's unchanged version, lines 296-327) with a `variance_adjust` branch taken before the existing `normalized` branch:

```python
    (
        u_node_counts, v_node_counts, u_total_count, v_total_count, tree_index,
    ) = _setup_pairwise_unifrac(
        u_counts, v_counts, taxa, tree, validate, normalized=normalized, unweighted=False,
    )
    branch_lengths = tree_index["length"]

    if variance_adjust:
        if not NUMBA_AVAILABLE:
            raise ImportError(
                "variance_adjust=True for weighted_unifrac requires numba."
            )
        return _weighted_unifrac_vaw(
            u_node_counts, v_node_counts, u_total_count, v_total_count,
            branch_lengths, normalized,
        )
    if normalized:
        ...  # unchanged existing branch
```

Add `variance_adjust=False` to the public signature at line 169 (after `normalized`, before `validate`).

- [ ] **Step 4: Extend the numba pdist kernel**

In `_weighted_unifrac_row_nb` (lines 652-700), add a `variance_adjust` parameter; when true, after computing `up`/`vp` per node inside the `k` loop, compute `m = sample_totals[row] + sample_totals[j]`, `mi = counts_by_node[row, k] + counts_by_node[j, k]`, `vaw = sqrt(mi*(m-mi))`, skip the node if `vaw <= 0`, else use `branch_lengths[k] / vaw` in place of `branch_lengths[k]` for both the `wu` and `c` accumulations (mirroring Step 3's math exactly). Thread `variance_adjust` through `_weighted_unifrac_pdist_nb` and `_weighted_unifrac_pdist_numba` the same way `normalized` is already threaded.

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac.py -k "weighted_unifrac" -v
```
Expected: PASS, including all pre-existing `weighted_unifrac` tests.

- [ ] **Step 6: Commit**

```bash
git add skbio/diversity/beta/_unifrac.py skbio/diversity/beta/tests/test_unifrac.py
git commit -m "feat: add variance_adjust to weighted_unifrac"
```

---

## Task 4: CPU — new `generalized_unifrac` function

**Files:**
- Modify: `skbio/diversity/beta/_unifrac.py` (new public function + new private kernel, placed after `weighted_unifrac`'s block)
- Modify: `skbio/diversity/beta/__init__.py` (export `generalized_unifrac`)
- Test: `skbio/diversity/beta/tests/test_unifrac.py`

**Interfaces:**
- Produces: `generalized_unifrac(u_counts, v_counts, taxa, tree, alpha=1.0, variance_adjust=False, validate=True) -> float`. Numba-only, raises `ImportError` if numba unavailable (always, not just when `variance_adjust=True`, since this function has no cython fallback at all).
- Consumes: `_setup_pairwise_unifrac` (existing, reused with `unweighted=False`).

- [ ] **Step 1: Write the failing tests**

```python
from skbio.diversity.beta import unweighted_unifrac, weighted_unifrac, generalized_unifrac

def test_generalized_unifrac_matches_ssu_fixture(self):
    table, taxa, tree, sample_ids = self._load_qiime_191_tt()
    for alpha, fname in [(0.5, 'generalized_unifrac_alpha0.5_dm.txt'),
                         (1.0, 'generalized_unifrac_alpha1.0_dm.txt')]:
        expected = self._load_dm_fixture(fname)
        for i, j in [(0, 1), (2, 5), (3, 7)]:
            obs = generalized_unifrac(table[i], table[j], taxa, tree, alpha=alpha)
            self.assertAlmostEqual(
                obs, expected[sample_ids[i], sample_ids[j]], places=5
            )

def test_generalized_unifrac_variance_adjust_matches_ssu_fixture(self):
    table, taxa, tree, sample_ids = self._load_qiime_191_tt()
    expected = self._load_dm_fixture('generalized_unifrac_alpha1.0_vaw_dm.txt')
    for i, j in [(0, 1), (2, 5), (3, 7)]:
        obs = generalized_unifrac(
            table[i], table[j], taxa, tree, alpha=1.0, variance_adjust=True
        )
        self.assertAlmostEqual(obs, expected[sample_ids[i], sample_ids[j]], places=5)

def test_generalized_unifrac_alpha_out_of_range_raises(self):
    with self.assertRaises(ValueError):
        generalized_unifrac(
            [1, 0, 1], [0, 1, 1], ['a', 'b', 'c'], self.t1, alpha=1.5
        )

def test_generalized_unifrac_both_empty_is_zero(self):
    obs = generalized_unifrac([0, 0, 0], [0, 0, 0], ['a', 'b', 'c'], self.t1)
    self.assertEqual(obs, 0.0)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac.py -k "generalized_unifrac" -v
```
Expected: FAIL — `ImportError: cannot import name 'generalized_unifrac'`.

- [ ] **Step 3: Implement the per-pair formula**

```python
def _generalized_unifrac(
    u_node_counts, v_node_counts, u_total_count, v_total_count, branch_lengths,
    alpha, variance_adjust,
):
    if u_total_count > 0:
        up = u_node_counts / u_total_count
    else:
        up = u_node_counts.astype(np.float64)
    if v_total_count > 0:
        vp = v_node_counts / v_total_count
    else:
        vp = v_node_counts.astype(np.float64)
    s = up + vp
    d = np.abs(up - vp)
    if variance_adjust:
        m = u_total_count + v_total_count
        mi = u_node_counts.astype(np.float64) + v_node_counts
        with np.errstate(invalid="ignore"):
            vaw = np.sqrt(mi * (m - mi))
        mask = (vaw > 0) & (s != 0.0)
        safe_vaw = np.where(vaw > 0, vaw, 1.0)
        s = s / safe_vaw
        d = d / safe_vaw
    else:
        mask = s != 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        sum_pow = np.where(mask, branch_lengths * np.power(s, alpha), 0.0)
        numerator = np.where(mask, sum_pow * (d / np.where(s != 0, s, 1.0)), 0.0).sum()
    denominator = sum_pow.sum()
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
```

- [ ] **Step 4: Implement the public function**

```python
@params_aliased([("taxa", "otu_ids", "0.6.0", True)])
def generalized_unifrac(
    u_counts, v_counts, taxa, tree, alpha=1.0, variance_adjust=False, validate=True,
):
    """Compute generalized UniFrac (GUniFrac).

    Parameters
    ----------
    u_counts, v_counts : list, np.array
        Vectors of counts/abundances of taxa for two samples.
    taxa : list, np.array
        Vector of taxon IDs corresponding to tip names in ``tree``.
    tree : TreeNode
        Tree relating taxa.
    alpha : float, optional
        GUniFrac alpha parameter in ``[0, 1]``, default 1.0.
    variance_adjust : bool, optional
        If ``True``, apply variance adjustment (VAW-UniFrac).
    validate : bool, optional
        If ``False``, skip input validation.

    Returns
    -------
    float
        The generalized UniFrac distance between the two samples.

    Raises
    ------
    ImportError
        If numba is not installed (this function has no cython path).
    ValueError
        If ``alpha`` is outside ``[0, 1]``.

    References
    ----------
    .. [1] Chen, J. et al. Associating microbiome composition with
       environmental covariates using generalized UniFrac distances.
       Bioinformatics 28, 2106-2113 (2012).

    """
    if not NUMBA_AVAILABLE:
        raise ImportError("generalized_unifrac requires numba.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    (
        u_node_counts, v_node_counts, u_total_count, v_total_count, tree_index,
    ) = _setup_pairwise_unifrac(
        u_counts, v_counts, taxa, tree, validate, normalized=True, unweighted=False,
    )
    if u_total_count == 0.0 and v_total_count == 0.0:
        return 0.0
    return _generalized_unifrac(
        u_node_counts, v_node_counts, u_total_count, v_total_count,
        tree_index["length"], alpha, variance_adjust,
    )
```

- [ ] **Step 5: Add the numba pdist kernel**

Mirror the structure of `_weighted_unifrac_row_nb`/`_weighted_unifrac_pdist_nb` (lines 652-789), replacing the inner accumulation with the Step 3 formula written as scalar numba code (no `np.where`, plain `if`/`for` inside the `njit` function, matching the existing kernels' style):

```python
@njit(inline="always")
def _generalized_unifrac_row_nb(
    row, n_samples, n_nodes, counts_by_node, branch_lengths, sample_totals,
    alpha, variance_adjust, out,
):
    base = _condensed_row_base(row, n_samples)
    row_total = sample_totals[row]
    for j in range(row + 1, n_samples):
        v_total = sample_totals[j]
        numerator = 0.0
        denominator = 0.0
        for k in range(n_nodes):
            up = counts_by_node[row, k] / row_total if row_total > 0.0 else 0.0
            vp = counts_by_node[j, k] / v_total if v_total > 0.0 else 0.0
            s = up + vp
            d = abs(up - vp)
            if variance_adjust:
                m = row_total + v_total
                mi = counts_by_node[row, k] + counts_by_node[j, k]
                vaw = np.sqrt(mi * (m - mi))
                if vaw <= 0.0:
                    continue
                s = s / vaw
                d = d / vaw
            if s == 0.0:
                continue
            length = branch_lengths[k]
            sum_pow = length * s ** alpha
            numerator += sum_pow * (d / s)
            denominator += sum_pow
        idx = base + j
        out[idx] = 0.0 if denominator == 0.0 else numerator / denominator
```

Add `_generalized_unifrac_pdist_nb` (parallel, row+mirror, same shape as `_weighted_unifrac_pdist_nb`) and `_generalized_unifrac_pdist_numba(counts, taxa, tree, alpha, variance_adjust, validate)`, reusing `_setup_multiple_unifrac`/`_get_tip_indices`. This is the function `_setup_multiple_weighted_unifrac`-equivalent wiring will call for the `beta_diversity`/numba-pdist path added in Task 5.

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac.py -k "generalized_unifrac" -v
```
Expected: PASS.

- [ ] **Step 7: Export the new function**

In `skbio/diversity/beta/__init__.py`, add `generalized_unifrac` to the existing import from `._unifrac` and to `__all__` (grep the file first for the exact existing pattern used for `unweighted_unifrac`/`weighted_unifrac`).

- [ ] **Step 8: Commit**

```bash
git add skbio/diversity/beta/_unifrac.py skbio/diversity/beta/__init__.py skbio/diversity/beta/tests/test_unifrac.py
git commit -m "feat: add generalized_unifrac (GUniFrac)"
```

---

## Task 5: `beta_diversity()` wiring

**Files:**
- Modify: `skbio/diversity/_driver.py` (metric registry and `_UNIFRAC_FAST_ENGINE`-style dispatch, around lines 234-372 per the earlier grep)
- Test: `skbio/diversity/beta/tests/test_unifrac.py`, `skbio/diversity/tests/test_driver.py` (grep for the existing `beta_diversity("weighted_unifrac", ...)` test pattern before adding)

**Interfaces:**
- Produces: `beta_diversity("generalized_unifrac", counts, taxa=taxa, tree=tree, alpha=0.5)` and `beta_diversity("unweighted_unifrac", counts, taxa=taxa, tree=tree, variance_adjust=True)` working end to end.
- Consumes: `_generalized_unifrac_pdist_numba` (Task 4), `_unweighted_unifrac_pdist_numba`/`_weighted_unifrac_pdist_numba` (Tasks 2-3, now accepting `variance_adjust`).

- [ ] **Step 1: Write the failing test**

```python
# in an appropriate beta_diversity test module (grep for the existing
# "beta_diversity('weighted_unifrac'" call to find the right file/class)
def test_beta_diversity_generalized_unifrac(self):
    from skbio.diversity import beta_diversity
    table, taxa, tree, sample_ids = self._load_qiime_191_tt()
    dm = beta_diversity(
        "generalized_unifrac", table, ids=sample_ids, taxa=taxa, tree=tree, alpha=0.5,
    )
    expected = self._load_dm_fixture('generalized_unifrac_alpha0.5_dm.txt')
    self.assertAlmostEqual(
        dm['f2', 'f1'], expected['f2', 'f1'], places=5
    )

def test_beta_diversity_unweighted_unifrac_variance_adjust(self):
    from skbio.diversity import beta_diversity
    table, taxa, tree, sample_ids = self._load_qiime_191_tt()
    dm = beta_diversity(
        "unweighted_unifrac", table, ids=sample_ids, taxa=taxa, tree=tree,
        variance_adjust=True,
    )
    expected = self._load_dm_fixture('unweighted_unifrac_vaw_dm.txt')
    self.assertAlmostEqual(dm['f2', 'f1'], expected['f2', 'f1'], places=5)
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest -k "beta_diversity_generalized_unifrac or beta_diversity_unweighted_unifrac_variance_adjust" -v
```
Expected: FAIL — `generalized_unifrac` not a recognized metric, or `variance_adjust` not accepted by the `unweighted_unifrac` dispatch path.

- [ ] **Step 3: Add `variance_adjust` passthrough and register `generalized_unifrac`**

In `skbio/diversity/_driver.py`, change line 347 from:

```python
    if metric in ("unweighted_unifrac", "weighted_unifrac"):
        taxa, tree, kwargs = _get_phylogenetic_kwargs(kwargs, taxa)
```
to:
```python
    if metric in ("unweighted_unifrac", "weighted_unifrac", "generalized_unifrac"):
        taxa, tree, kwargs = _get_phylogenetic_kwargs(kwargs, taxa)
```

Change the `unweighted_unifrac` branch (lines 350-363) to extract and pass `variance_adjust`:

```python
    if metric == "unweighted_unifrac":
        variance_adjust = kwargs.pop("variance_adjust", False)
        resolved_engine = _resolve_engine(
            engine, ("cython", "numba"), fast=_UNIFRAC_FAST_ENGINE
        )
        if resolved_engine == "numba" and _numba_unifrac_fast_path_eligible(
            engine, pairwise_func, kwargs
        ):
            distances = _unweighted_unifrac_pdist_numba(
                counts, taxa=taxa, tree=tree, normalized=True,
                variance_adjust=variance_adjust, validate=validate,
            )
            return DistanceMatrix(distances, ids)
        metric, counts = _setup_multiple_unweighted_unifrac(
            counts, taxa=taxa, tree=tree, validate=validate
        )
```

Change the `weighted_unifrac` branch (lines 364-384) to extract and pass `variance_adjust`:

```python
    elif metric == "weighted_unifrac":
        normalized = kwargs.pop("normalized", _normalize_weighted_unifrac_by_default)
        variance_adjust = kwargs.pop("variance_adjust", False)
        resolved_engine = _resolve_engine(
            engine, ("cython", "numba"), fast=_UNIFRAC_FAST_ENGINE
        )
        if resolved_engine == "numba" and _numba_unifrac_fast_path_eligible(
            engine, pairwise_func, kwargs
        ):
            distances = _weighted_unifrac_pdist_numba(
                counts, taxa=taxa, tree=tree, normalized=normalized,
                variance_adjust=variance_adjust, validate=validate,
            )
            return DistanceMatrix(distances, ids)
        metric, counts = _setup_multiple_weighted_unifrac(
            counts, taxa=taxa, tree=tree, normalized=normalized, validate=validate
        )
```

Add a new branch immediately after it (before the existing `elif metric == "manhattan":` at line 385), matching the same structural shape but always numba (no cython fallback):

```python
    elif metric == "generalized_unifrac":
        if not NUMBA_AVAILABLE:
            raise ImportError("generalized_unifrac requires numba.")
        alpha = kwargs.pop("alpha", 1.0)
        variance_adjust = kwargs.pop("variance_adjust", False)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        distances = _generalized_unifrac_pdist_numba(
            counts, taxa=taxa, tree=tree, alpha=alpha,
            variance_adjust=variance_adjust, validate=validate,
        )
        return DistanceMatrix(distances, ids)
```

(Import `_generalized_unifrac_pdist_numba` alongside the existing `_unweighted_unifrac_pdist_numba`/`_weighted_unifrac_pdist_numba` imports near the top of `_driver.py` — grep that existing import line first and add to it.)

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest -k "beta_diversity_generalized_unifrac or beta_diversity_unweighted_unifrac_variance_adjust" -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add skbio/diversity/_driver.py skbio/diversity/beta/tests/test_unifrac.py
git commit -m "feat: wire generalized_unifrac and variance_adjust into beta_diversity"
```

---

## Task 6: GPU backend detection module

**Files:**
- Create: `skbio/diversity/beta/_unifrac_gpu.py`
- Modify: `pyproject.toml` or `setup.py` (grep which this repo uses) to add `gpu-nvidia`/`gpu-amd` optional extras (`numba-cuda`, `hip-python`)
- Test: `skbio/diversity/beta/tests/test_unifrac_gpu.py` (new file)

**Interfaces:**
- Produces: `detect_gpu_backend() -> Literal["cuda", "hip", None]`, `get_cuda_module()` (returns the active `numba.cuda`-compatible module, either real `numba.cuda` or `numba.hip` after `pose_as_cuda()`, raising `ImportError` with an actionable message if neither is importable when actually needed).
- Consumes: nothing from this codebase; wraps `numba.cuda`/`numba.hip` directly.

- [ ] **Step 1: Write the failing tests**

```python
# skbio/diversity/beta/tests/test_unifrac_gpu.py
from unittest import TestCase, main
from skbio.diversity.beta._unifrac_gpu import detect_gpu_backend, get_cuda_module


class GpuBackendTests(TestCase):

    def test_detect_gpu_backend_returns_known_value(self):
        self.assertIn(detect_gpu_backend(), ("cuda", "hip", None))

    def test_get_cuda_module_raises_cleanly_when_no_gpu(self):
        backend = detect_gpu_backend()
        if backend is not None:
            self.skipTest("a GPU backend is available in this environment")
        with self.assertRaises(ImportError):
            get_cuda_module()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac_gpu.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'skbio.diversity.beta._unifrac_gpu'`.

- [ ] **Step 3: Implement the module**

```python
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
```

(Verify `numba.hip`'s actual availability-check and `pose_as_cuda` call signatures against its README on whichever machine first installs it — the GPU feasibility research cited `github.com/ROCm/hip-python`'s `numba_hip` component as the source; confirm the exact import path and function names haven't shifted since that research before trusting this verbatim in Task 11/12.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac_gpu.py -v
```
Expected: PASS (on this machine, with neither `numba-cuda` nor `numba.hip` installed yet, `detect_gpu_backend()` returns `None` and the `ImportError` test runs).

- [ ] **Step 5: Add optional dependency extras**

```bash
grep -n "\[project.optional-dependencies\]" pyproject.toml
```
Add under that section (matching the file's existing style, checked via the grep above):
```toml
gpu-nvidia = ["numba-cuda"]
gpu-amd = ["hip-python"]
```

- [ ] **Step 6: Commit**

```bash
git add skbio/diversity/beta/_unifrac_gpu.py skbio/diversity/beta/tests/test_unifrac_gpu.py pyproject.toml
git commit -m "feat: add GPU backend detection for unifrac"
```

---

## Task 7: GPU kernel — weighted family

**Files:**
- Modify: `skbio/diversity/beta/_unifrac_gpu.py`
- Test: `skbio/diversity/beta/tests/test_unifrac_gpu.py`

**Interfaces:**
- Produces: `weighted_unifrac_gpu(counts_by_node, branch_lengths, sample_totals, normalized, variance_adjust) -> np.ndarray` (condensed), built on a shared `_build_pair_index(n_samples) -> (pair_i, pair_j)` helper and the method-enum kernel described below.
- Consumes: `get_cuda_module` (Task 6).

- [ ] **Step 1: Write the failing test** (skipped unless a GPU backend is present — this test is the one actually exercised in Tasks 11/12's dual-backend verification, not expected to pass on a GPU-less dev machine)

```python
from unittest import TestCase
import numpy as np

from skbio.diversity.beta._unifrac_gpu import detect_gpu_backend, weighted_unifrac_gpu
from skbio.diversity.beta.tests._fixtures import QiimeTinyTestMixin


class WeightedUnifracGpuTests(QiimeTinyTestMixin, TestCase):

    def setUp(self):
        if detect_gpu_backend() is None:
            self.skipTest("no GPU backend available")

    def test_weighted_unifrac_gpu_matches_cpu(self):
        from skbio.diversity.beta._unifrac import _weighted_unifrac_pdist_numba
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        gpu = weighted_unifrac_gpu(table, taxa, tree, normalized=False, validate=True)
        cpu = _weighted_unifrac_pdist_numba(table, taxa, tree, normalized=False, validate=True)
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)
```

(`QiimeTinyTestMixin` is the shared mixin created in Task 2 Step 1 — `test_unifrac_gpu.py`'s other test classes in Tasks 8-9 inherit it the same way, so `_load_qiime_191_tt`/`_load_dm_fixture` are never duplicated or redefined.)

- [ ] **Step 2: Run to verify it's skipped (no GPU on this dev machine) or fails (if a GPU is present)**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac_gpu.py -k weighted -v
```
Expected: SKIPPED on a GPU-less machine; `AttributeError`/`ImportError` on a GPU-present machine (function doesn't exist yet) — either confirms the test is wired correctly before Task 11/12 actually run it on real hardware.

- [ ] **Step 3: Implement the method-enum constants and pair-index helper**

```python
import numpy as np

UNWEIGHTED = 0
UNWEIGHTED_UNNORMALIZED = 1
WEIGHTED_NORMALIZED = 2
WEIGHTED_UNNORMALIZED = 3
GENERALIZED = 4


def _build_pair_index(n_samples):
    """Precompute condensed-index (i, j) pairs for i < j, scipy pdist order."""
    pair_i = np.empty(n_samples * (n_samples - 1) // 2, dtype=np.int32)
    pair_j = np.empty_like(pair_i)
    idx = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pair_i[idx] = i
            pair_j[idx] = j
            idx += 1
    return pair_i, pair_j
```

- [ ] **Step 4: Implement the shared GPU kernel**

```python
def _make_unifrac_kernel(cuda):
    """Build the UniFrac pair kernel against the given cuda-API module."""

    @cuda.jit
    def _unifrac_pair_kernel(
        proportions, counts, sample_totals, branch_lengths, method, alpha,
        variance_adjust, pair_i, pair_j, out,
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
                    numerator += length if differs else 0.0
                    denominator += length
            elif method == GENERALIZED:
                if s == 0.0:
                    continue
                sum_pow = length * s ** alpha
                numerator += sum_pow * (d / s)
                denominator += sum_pow
        if method == WEIGHTED_UNNORMALIZED or method == UNWEIGHTED_UNNORMALIZED:
            out[idx] = numerator
        else:
            out[idx] = 0.0 if denominator == 0.0 else numerator / denominator

    return _unifrac_pair_kernel
```

Note: import `math` at module top (the `math.sqrt`/`**` ops above must use `math`, not `numpy`, inside a `cuda.jit` kernel body — numba's CUDA target supports `math` module functions, not general numpy ufuncs, inside device code).

- [ ] **Step 5: Implement the host-side driver function**

```python
def weighted_unifrac_gpu(counts, taxa, tree, normalized, variance_adjust, validate=True):
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
        counts_by_node, sample_totals[:, None],
        out=np.zeros_like(counts_by_node), where=sample_totals[:, None] > 0,
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
        d_proportions, d_counts, d_sample_totals, d_branch_lengths,
        method, 1.0, variance_adjust, d_pair_i, d_pair_j, d_out,
    )
    return d_out.copy_to_host()
```

Note this uses plain `branch_lengths` summed over every node for the normalized case (per the Global Constraints note), which is **not** numerically identical to the existing CPU `weighted_unifrac(normalized=True)`'s `node_to_root_distances`-based result for non-VAW cases unless that tip-to-root identity is confirmed to hold exactly — this is exactly what Step 1's `test_weighted_unifrac_gpu_matches_cpu` checks, and if it fails, that identity needs re-deriving rather than assumed, don't skip past a failure here by loosening the tolerance.

- [ ] **Step 6: Run tests** (on a GPU-backed machine — this step is realistically executed as part of Task 11/12, noted here for completeness of the task's own test cycle)

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac_gpu.py -k weighted -v
```
Expected: PASS once run on actual GPU hardware (Task 11 or 12).

- [ ] **Step 7: Commit**

```bash
git add skbio/diversity/beta/_unifrac_gpu.py skbio/diversity/beta/tests/test_unifrac_gpu.py skbio/diversity/beta/tests/_fixtures.py
git commit -m "feat: add GPU-agnostic kernel for weighted unifrac"
```

---

## Task 8: GPU kernel — unweighted family

**Files:**
- Modify: `skbio/diversity/beta/_unifrac_gpu.py`
- Test: `skbio/diversity/beta/tests/test_unifrac_gpu.py`

**Interfaces:**
- Produces: `unweighted_unifrac_gpu(counts, taxa, tree, normalized, variance_adjust, validate=True) -> np.ndarray` (condensed).
- Consumes: `_make_unifrac_kernel`, `_build_pair_index`, `get_cuda_module` (Task 7/6) — the shared kernel from Task 7 already has the `UNWEIGHTED`/`UNWEIGHTED_UNNORMALIZED` branch implemented, this task only adds the host-side driver and tests.

- [ ] **Step 1: Write the failing test**

```python
from unittest import TestCase
import numpy as np

from skbio.diversity.beta._unifrac_gpu import detect_gpu_backend, unweighted_unifrac_gpu
from skbio.diversity.beta.tests._fixtures import QiimeTinyTestMixin


class UnweightedUnifracGpuTests(QiimeTinyTestMixin, TestCase):

    def setUp(self):
        if detect_gpu_backend() is None:
            self.skipTest("no GPU backend available")

    def test_unweighted_unifrac_gpu_matches_cpu(self):
        from skbio.diversity.beta._unifrac import _unweighted_unifrac_pdist_numba
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        gpu = unweighted_unifrac_gpu(table, taxa, tree, normalized=True, variance_adjust=False)
        cpu = _unweighted_unifrac_pdist_numba(
            table, taxa, tree, normalized=True, variance_adjust=False, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)
```

- [ ] **Step 2: Run to verify failure/skip** (same pattern as Task 7 Step 2)

- [ ] **Step 3: Implement the host-side driver**

```python
def unweighted_unifrac_gpu(counts, taxa, tree, normalized, variance_adjust, validate=True):
    from skbio.diversity.beta._unifrac import _setup_multiple_unifrac, _get_tip_indices
    cuda = get_cuda_module()
    counts_by_node, tree_index, branch_lengths = _setup_multiple_unifrac(
        counts, taxa, tree, validate
    )
    counts_by_node = np.ascontiguousarray(counts_by_node, dtype=np.float64)
    tip_indices = _get_tip_indices(tree_index)
    sample_totals = counts_by_node[:, tip_indices].sum(axis=1)
    n_samples = counts_by_node.shape[0]
    # presence/absence kernel branch reads proportions only to test > 0,
    # so pass raw counts as "proportions" directly, no division needed.
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
        d_proportions, d_counts, d_sample_totals, d_branch_lengths,
        method, 1.0, variance_adjust, d_pair_i, d_pair_j, d_out,
    )
    return d_out.copy_to_host()
```

- [ ] **Step 4: Run tests** (GPU hardware, Task 11/12)

- [ ] **Step 5: Commit**

```bash
git add skbio/diversity/beta/_unifrac_gpu.py skbio/diversity/beta/tests/test_unifrac_gpu.py
git commit -m "feat: add GPU-agnostic kernel driver for unweighted unifrac"
```

---

## Task 9: GPU kernel — generalized

**Files:**
- Modify: `skbio/diversity/beta/_unifrac_gpu.py`
- Test: `skbio/diversity/beta/tests/test_unifrac_gpu.py`

**Interfaces:**
- Produces: `generalized_unifrac_gpu(counts, taxa, tree, alpha, variance_adjust, validate=True) -> np.ndarray` (condensed).
- Consumes: Task 7's shared kernel (`GENERALIZED` branch already implemented there).

- [ ] **Step 1: Write the failing test**

```python
from unittest import TestCase
import numpy as np

from skbio.diversity.beta._unifrac_gpu import detect_gpu_backend, generalized_unifrac_gpu
from skbio.diversity.beta.tests._fixtures import QiimeTinyTestMixin


class GeneralizedUnifracGpuTests(QiimeTinyTestMixin, TestCase):

    def setUp(self):
        if detect_gpu_backend() is None:
            self.skipTest("no GPU backend available")

    def test_generalized_unifrac_gpu_matches_cpu(self):
        from skbio.diversity.beta._unifrac import _generalized_unifrac_pdist_numba
        table, taxa, tree, sample_ids = self._load_qiime_191_tt()
        gpu = generalized_unifrac_gpu(table, taxa, tree, alpha=0.5, variance_adjust=False)
        cpu = _generalized_unifrac_pdist_numba(
            table, taxa, tree, alpha=0.5, variance_adjust=False, validate=True
        )
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)
```

- [ ] **Step 2: Run to verify failure/skip**

- [ ] **Step 3: Implement the host-side driver** (same structure as Task 7's `weighted_unifrac_gpu`, `method=GENERALIZED`, passing the real `alpha` instead of the hardcoded `1.0`)

```python
def generalized_unifrac_gpu(counts, taxa, tree, alpha, variance_adjust, validate=True):
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
        counts_by_node, sample_totals[:, None],
        out=np.zeros_like(counts_by_node), where=sample_totals[:, None] > 0,
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
        d_proportions, d_counts, d_sample_totals, d_branch_lengths,
        GENERALIZED, alpha, variance_adjust, d_pair_i, d_pair_j, d_out,
    )
    return d_out.copy_to_host()
```

- [ ] **Step 4: Run tests** (GPU hardware, Task 11/12)

- [ ] **Step 5: Commit**

```bash
git add skbio/diversity/beta/_unifrac_gpu.py skbio/diversity/beta/tests/test_unifrac_gpu.py
git commit -m "feat: add GPU-agnostic kernel driver for generalized unifrac"
```

---

## Task 10: Public API — GPU engine wiring

**Files:**
- Modify: `skbio/diversity/beta/_unifrac.py` (`unweighted_unifrac`, `weighted_unifrac`, `generalized_unifrac` — add `engine` parameter)
- Modify: `skbio/diversity/_driver.py` (`beta_diversity`'s `engine` dispatch)
- Test: `skbio/diversity/beta/tests/test_unifrac.py`

**Interfaces:**
- Produces: `engine='gpu'` accepted alongside the existing `{'cython', 'numba', 'fast'}` for `unweighted_unifrac`/`weighted_unifrac`, and for `generalized_unifrac` (which gets its own `engine={'numba', 'gpu'}` since it has no cython path). Requesting `engine='gpu'` with no GPU detected raises a clear error; `engine='fast'`/default silently uses CPU numba when no GPU is present.

- [ ] **Step 1: Write the failing tests**

```python
def test_unweighted_unifrac_gpu_engine_raises_without_gpu(self):
    from skbio.diversity.beta._unifrac_gpu import detect_gpu_backend
    if detect_gpu_backend() is not None:
        self.skipTest("a GPU backend is available, this test checks the no-GPU error path")
    with self.assertRaises(ImportError):
        unweighted_unifrac(
            [1, 0, 1], [0, 1, 1], ['a', 'b', 'c'], self.t1, engine='gpu'
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac.py -k gpu_engine_raises -v
```
Expected: FAIL — `unweighted_unifrac() got an unexpected keyword argument 'engine'`.

- [ ] **Step 3: Add `engine` parameter to the three public functions**

For each of `unweighted_unifrac`, `weighted_unifrac`, `generalized_unifrac`, add `engine=None` to the signature (after `variance_adjust`, before `validate`) and, at the top of the function body, resolve it:

```python
    if engine == "gpu":
        from skbio.diversity.beta._unifrac_gpu import detect_gpu_backend
        if detect_gpu_backend() is None:
            raise ImportError(
                "engine='gpu' was requested but no usable GPU backend "
                "(numba-cuda or numba.hip) was found."
            )
        # single-pair GPU dispatch is wasteful (kernel launch overhead for
        # one pair); route through the same GPU pdist function used by
        # beta_diversity with a 2-row input, documented as such rather
        # than hidden.
        from skbio.diversity.beta._unifrac_gpu import weighted_unifrac_gpu  # (etc. per function)
        ...
```

(Exact per-function wiring mirrors each function's existing `_setup_pairwise_unifrac`/kernel call, substituting a 2-row call into the matching `*_gpu` driver from Tasks 7-9 instead of the CPU kernel when `engine == "gpu"`.)

- [ ] **Step 4: Add GPU dispatch to `beta_diversity`**

In `_driver.py`, insert a GPU check before each existing `resolved_engine = _resolve_engine(...)` call added in Task 5. For the `unweighted_unifrac` branch:

```python
    if metric == "unweighted_unifrac":
        variance_adjust = kwargs.pop("variance_adjust", False)
        if engine == "gpu":
            from skbio.diversity.beta._unifrac_gpu import (
                detect_gpu_backend, unweighted_unifrac_gpu,
            )
            if detect_gpu_backend() is None:
                raise ImportError(
                    "engine='gpu' was requested but no usable GPU backend was found."
                )
            distances = unweighted_unifrac_gpu(
                counts, taxa, tree, normalized=True,
                variance_adjust=variance_adjust, validate=validate,
            )
            return DistanceMatrix(distances, ids)
        resolved_engine = _resolve_engine(
            engine, ("cython", "numba"), fast=_UNIFRAC_FAST_ENGINE
        )
        ...  # unchanged from Task 5
```

Apply the same `if engine == "gpu": ... ; return DistanceMatrix(...)` guard, before the existing `_resolve_engine` call, to the `weighted_unifrac` branch (calling `weighted_unifrac_gpu(counts, taxa, tree, normalized=normalized, variance_adjust=variance_adjust, validate=validate)`) and to the `generalized_unifrac` branch added in Task 5 (calling `generalized_unifrac_gpu(counts, taxa, tree, alpha=alpha, variance_adjust=variance_adjust, validate=validate)`, replacing its unconditional `NUMBA_AVAILABLE` check with: try GPU first if `engine == "gpu"`, else require numba as already written).

Note `engine='fast'`/`None` (the default) never routes to GPU automatically — matching the spec's "requesting the GPU engine with no GPU backend detected raises... when explicitly requested; falls back silently to CPU only when the caller asked scikit-bio to choose automatically": only an explicit `engine='gpu'` takes this path, so there's no `_numba_unifrac_fast_path_eligible`-style auto-selection to add for it.

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac.py -k gpu -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add skbio/diversity/beta/_unifrac.py skbio/diversity/_driver.py skbio/diversity/beta/tests/test_unifrac.py
git commit -m "feat: wire engine='gpu' into unifrac public API and beta_diversity"
```

---

## Task 11: NRP (NVIDIA) end-to-end verification

**Revised 2026-10-02:** no custom container image, no registry. NRP is just Kubernetes — use a public NVIDIA CUDA base image and install everything (scikit-bio included) at pod startup via pip/git, so nothing needs pushing anywhere and no registry credentials are needed inside or outside the pod. The scikit-bio branch must be reachable via a public (or at least pip-installable) git URL for this to work — push the current branch to the `fork` remote (`git@github.com:l1joseph/scikit-bio.git`) first if it isn't already there.

**Files:**
- Create (dev-only, scratch): `/cosmos/vast/scratch/l1joseph/unifrac-bench/nrp/job.yaml`

**Interfaces:**
- Produces: a passing run on an NRP A100 node confirming `weighted_unifrac_gpu`/`unweighted_unifrac_gpu`/`generalized_unifrac_gpu` (Tasks 7-9) match all 11 `ssu` fixtures (Task 1) within a measured tolerance, on real NVIDIA hardware via `numba-cuda`.

- [ ] **Step 1: Push the branch to the public fork**

```bash
cd /cosmos/nfs/home/l1joseph/scikit-bio/.claude/worktrees/gpu-agnostic-unifrac
git push fork worktree-gpu-agnostic-unifrac
```
Confirm it's actually reachable publicly: `curl -sL https://raw.githubusercontent.com/l1joseph/scikit-bio/worktree-gpu-agnostic-unifrac/pyproject.toml | head -5` should print real file content, not a 404 or an auth-wall page. If the fork is private, this step blocks here — report BLOCKED rather than guessing around it (don't fall back to a registry/image approach without checking with the controller first, that's explicitly what this revision is trying to avoid).

- [ ] **Step 2: Write the Kubernetes job manifest**

Modeled on the real, existing `knightlab-ml` namespace job pattern (`oceanpredict-st10-train-a100-001`, confirmed via `kubectl get job ... -o yaml` during planning). The container installs numba-cuda and the scikit-bio branch at startup, then runs an inline Python verification script via a heredoc — no image build, no `COPY`, no registry:

```yaml
# nrp/job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: unifrac-gpu-verify-001
  namespace: knightlab-ml
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: ["NVIDIA-A100-SXM4-40GB", "NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe"]
      containers:
      - name: verify
        image: nvidia/cuda:12.4.1-devel-ubuntu22.04
        resources:
          limits:
            cpu: "2"
            memory: 8Gi
            nvidia.com/a100: "1"
          requests:
            cpu: "2"
            memory: 8Gi
            nvidia.com/a100: "1"
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -euo pipefail
            apt-get update -qq && apt-get install -y -qq python3-pip git > /dev/null
            pip install -q numba numba-cuda
            pip install -q "git+https://github.com/l1joseph/scikit-bio.git@worktree-gpu-agnostic-unifrac"
            python3 - <<'PYEOF'
            import numpy as np
            import pandas as pd
            from skbio import TreeNode, DistanceMatrix
            from skbio.util._testing import get_data_path
            from skbio.diversity.beta._unifrac_gpu import (
                detect_gpu_backend, weighted_unifrac_gpu, unweighted_unifrac_gpu,
                generalized_unifrac_gpu,
            )

            backend = detect_gpu_backend()
            print(f"Detected GPU backend: {backend}", flush=True)
            assert backend == "cuda", f"expected cuda backend on NRP, got {backend}"

            base = get_data_path('qiime-191-tt', subfolder='data')
            df = pd.read_csv(f'{base}/otu-table.tsv', sep='\t', skiprows=1, index_col=0)
            taxa = df.index.astype(str).tolist()
            table = df.T.values
            tree = TreeNode.read(f'{base}/tree.nwk')

            checks = [
                ("weighted_unnormalized", weighted_unifrac_gpu(table, taxa, tree, False, False), "weighted_unifrac_dm.txt"),
                ("weighted_normalized", weighted_unifrac_gpu(table, taxa, tree, True, False), "weighted_normalized_unifrac_dm.txt"),
                ("weighted_unnormalized_vaw", weighted_unifrac_gpu(table, taxa, tree, False, True), "weighted_unifrac_vaw_dm.txt"),
                ("weighted_normalized_vaw", weighted_unifrac_gpu(table, taxa, tree, True, True), "weighted_normalized_unifrac_vaw_dm.txt"),
                ("unweighted", unweighted_unifrac_gpu(table, taxa, tree, True, False), "unweighted_unifrac_dm.txt"),
                ("unweighted_unnormalized", unweighted_unifrac_gpu(table, taxa, tree, False, False), "unweighted_unnormalized_unifrac_dm.txt"),
                ("unweighted_vaw", unweighted_unifrac_gpu(table, taxa, tree, True, True), "unweighted_unifrac_vaw_dm.txt"),
                ("unweighted_unnormalized_vaw", unweighted_unifrac_gpu(table, taxa, tree, False, True), "unweighted_unnormalized_unifrac_vaw_dm.txt"),
                ("generalized_alpha0.5", generalized_unifrac_gpu(table, taxa, tree, 0.5, False), "generalized_unifrac_alpha0.5_dm.txt"),
                ("generalized_alpha1.0", generalized_unifrac_gpu(table, taxa, tree, 1.0, False), "generalized_unifrac_alpha1.0_dm.txt"),
                ("generalized_alpha1.0_vaw", generalized_unifrac_gpu(table, taxa, tree, 1.0, True), "generalized_unifrac_alpha1.0_vaw_dm.txt"),
            ]

            max_dev = 0.0
            for name, condensed, fixture_name in checks:
                expected_dm = DistanceMatrix.read(f'{base}/{fixture_name}')
                expected_condensed = expected_dm.condensed_form()
                dev = np.abs(condensed - expected_condensed).max()
                max_dev = max(max_dev, dev)
                print(f"{name}: max abs deviation = {dev:.3e}", flush=True)

            print(f"OVERALL max abs deviation across all methods: {max_dev:.3e}", flush=True)
            assert max_dev < 1e-6, f"deviation {max_dev} exceeds threshold"
            print("ALL CHECKS PASSED", flush=True)
            PYEOF
```

Before trusting `get_data_path(...)` here, confirm it actually resolves correctly for a plain (non-editable) `pip install git+...` — it's primarily used in-repo for editable/test checkouts; read `skbio/util/_testing.py`'s implementation and verify it resolves paths relative to the installed package's own `__file__`, not something that only works in a source checkout. If it doesn't work this way, adapt (e.g. construct the path via `importlib.resources` relative to `skbio.diversity.beta.tests` directly) rather than assuming.

- [ ] **Step 3: Submit and check the job**

```bash
kubectl apply -f /cosmos/vast/scratch/l1joseph/unifrac-bench/nrp/job.yaml
kubectl wait --for=condition=complete --timeout=900s job/unifrac-gpu-verify-001 -n knightlab-ml || \
  kubectl logs job/unifrac-gpu-verify-001 -n knightlab-ml
kubectl logs job/unifrac-gpu-verify-001 -n knightlab-ml | tee /cosmos/vast/scratch/l1joseph/unifrac-bench/nrp/nrp-verify.log
```
Expected: log output ending in `ALL CHECKS PASSED`, with all 11 `max abs deviation` lines and the printed `OVERALL` figure. Allow extra time on the first run for `apt-get`/`pip install` inside the pod (no image caching since there's no custom image).

- [ ] **Step 4: Record the measured tolerance and clean up**

Copy the printed `OVERALL max abs deviation` value and all 11 per-method figures into this plan's Task 13 (tolerance finalization) and into the spec, then delete the job:

```bash
kubectl delete job unifrac-gpu-verify-001 -n knightlab-ml
```

---

## Task 12: Cosmos (AMD) end-to-end verification via `pose_as_cuda`

**Files:**
- Create (dev-only, scratch): `/cosmos/vast/scratch/l1joseph/unifrac-bench/amd/verify.py` (same content as Task 11 Step 2's `verify.py`, path-adjusted, since the kernel code is identical by construction — this is the actual proof of "one kernel, two backends")
- Create (dev-only, scratch): `/cosmos/vast/scratch/l1joseph/unifrac-bench/amd/run_verify.sh`

**Interfaces:**
- Produces: a passing run on a Cosmos MI300A node confirming the same kernels match the same fixtures via `numba.hip`'s `pose_as_cuda()`, verified natively on this cluster's own SLURM scheduler.

- [ ] **Step 1: Install `numba.hip`/`hip-python` into a scratch conda env**

```bash
mamba create -n unifrac-gpu-amd python=3.11 numpy biom-format pandas -y
conda run -n unifrac-gpu-amd pip install --index-url https://test.pypi.org/simple/ hip-python
# confirm the exact package/extras name against github.com/ROCm/hip-python's
# current README before trusting this verbatim — it was pre-1.0 and fast-moving
# per the GPU feasibility research; this step's success criterion is Step 3's
# import check, not this command's exit code alone.
```

- [ ] **Step 2: Copy the verify script**

```bash
mkdir -p /cosmos/vast/scratch/l1joseph/unifrac-bench/amd
cp /cosmos/vast/scratch/l1joseph/unifrac-bench/nrp/verify.py /cosmos/vast/scratch/l1joseph/unifrac-bench/amd/verify.py
# edit the sys.path.insert line to point at the real scikit-bio checkout
# instead of the Docker-image path, and remove the `assert backend == "cuda"`
# line, replacing it with `assert backend == "hip"`.
```

- [ ] **Step 3: Write and submit the srun verification script**

```bash
cat > /cosmos/vast/scratch/l1joseph/unifrac-bench/amd/run_verify.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=unifrac-gpu-verify-amd
#SBATCH --output=/cosmos/vast/scratch/l1joseph/unifrac-bench/amd/logs/%x_%j.out
#SBATCH --error=/cosmos/vast/scratch/l1joseph/unifrac-bench/amd/logs/%x_%j.err
#SBATCH --partition=cluster
#SBATCH --gpus=1
#SBATCH --time=00:30:00
mkdir -p /cosmos/vast/scratch/l1joseph/unifrac-bench/amd/logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate unifrac-gpu-amd
python /cosmos/vast/scratch/l1joseph/unifrac-bench/amd/verify.py
EOF
```

Show this script's contents to the user and confirm before submitting (per the standing convention that `sbatch` scripts are shown and confirmed first):

```bash
cat /cosmos/vast/scratch/l1joseph/unifrac-bench/amd/run_verify.sh
```

- [ ] **Step 4: Submit and check**

```bash
sbatch /cosmos/vast/scratch/l1joseph/unifrac-bench/amd/run_verify.sh
squeue -u $USER
# once completed:
cat /cosmos/vast/scratch/l1joseph/unifrac-bench/amd/logs/unifrac-gpu-verify-amd_*.out
```
Expected: log output ending in `ALL CHECKS PASSED`, with its own printed `OVERALL max abs deviation` figure (compare against Task 11's NVIDIA figure — they need not be identical, both just need to be within the eventual test tolerance, but a large discrepancy between the two backends' own deviation from the `ssu` fixtures would be a real finding worth investigating, not glossing over).

---

## Task 13: Tolerance finalization and docs

**Files:**
- Modify: `skbio/diversity/beta/tests/test_unifrac.py` (replace the `places=5` placeholders used throughout Tasks 2-5 with the actual measured tolerance)
- Modify: `CHANGELOG.md`
- Modify: `docs/superpowers/specs/2026-10-02-ssu-unifrac-numba-phase1-design.md` (record final measured tolerance)

**Interfaces:** None — this task finalizes numbers, it doesn't add new code surface.

- [ ] **Step 1: Collect the measured deviations**

Pull together: Task 1's CPU-vs-`ssu`-fixture deviations (observed while Tasks 2-4's tests were made to pass), and Task 11/12's GPU-vs-fixture deviations. Take the largest of all of them as the basis for the committed test tolerance, with a safety margin (e.g. 10x), not just whichever number happened to be printed last.

- [ ] **Step 2: Update test assertions with the real tolerance**

Replace every `self.assertAlmostEqual(obs, expected, places=5)` added in Tasks 2-5 with an explicit `delta=<measured_tolerance>` argument instead of `places=5`, documenting the measured figure in a comment above each, e.g.:

```python
# measured max deviation across all methods/backends: 3.1e-9 (see spec);
# 1e-7 gives a 30x safety margin
self.assertAlmostEqual(obs, expected[...], delta=1e-7)
```

- [ ] **Step 3: Run the full test suite**

```bash
python -m pytest skbio/diversity/beta/tests/test_unifrac.py skbio/diversity/beta/tests/test_unifrac_gpu.py -v
```
Expected: PASS.

- [ ] **Step 4: Update CHANGELOG.md**

Add an entry under the unreleased section (grep the file's existing most recent entries for exact formatting first) describing: new `generalized_unifrac` function, `variance_adjust` on `unweighted_unifrac`/`weighted_unifrac`, new `engine='gpu'` option backed by `numba-cuda`/`numba.hip`, new optional `gpu-nvidia`/`gpu-amd` extras.

- [ ] **Step 5: Finalize the spec with measured numbers**

Replace the spec's "measured once implemented and documented with the actual figure" placeholder language with the real numbers from Step 1.

- [ ] **Step 6: Commit**

```bash
git add skbio/diversity/beta/tests/test_unifrac.py skbio/diversity/beta/tests/test_unifrac_gpu.py CHANGELOG.md docs/superpowers/specs/2026-10-02-ssu-unifrac-numba-phase1-design.md
git commit -m "test: finalize measured tolerances for unifrac gpu/vaw correctness tests"
```
