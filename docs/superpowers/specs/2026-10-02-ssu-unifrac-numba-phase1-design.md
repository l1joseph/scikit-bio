# GPU-agnostic UniFrac: a dual-backend numba kernel for SSU's 5 methods

Status: approved design, pending implementation plan
Date: 2026-10-02 (revised from an earlier CPU-only draft of this spec)

## Background

Igor Sfiligoi and Qiyun Zhu both signed off on porting the C++ "SSU"
UniFrac implementation (`github.com/biocore/unifrac-binaries`) into
Numba, explicitly *not* building on scikit-bio's existing from-scratch
numba UniFrac kernel (#2558). Scoping work (source reading, dynamic CPU
profiling, GPU feasibility research, formula extraction, all recorded
in the `project_ssu_unifrac_numba_port` memory) went through several
corrections before landing here:

1. CPU profiling showed SSU's own CPU wall time is dominated by
   OpenMP/BLAS thread-pool startup, not its kernel, at small/medium
   scale. Reading skbio's *existing* numba kernel
   (`skbio/diversity/beta/_unifrac.py:528-789`) then showed it's
   already a reasonable, proven, condensed/row+mirror-parallelized CPU
   design — SSU's stripe/batching structure is a GPU-specific
   workaround for memory/register pressure that has no demonstrated
   CPU benefit over what's already there.
2. **That reframes the actual goal.** Igor's original claim was about
   SSU's GPU acceleration specifically ("UniFrac is already fully on
   GPU... the one in skbio is worthless"). The CPU kernel is not where
   the real gap is. **GPU support is the primary goal of this work**,
   not a later phase after a CPU rewrite.
3. SSU genuinely supports 5 distinct methods (verified against the
   real `ssu` binary, not just source reading): `unweighted`,
   `unweighted_unnormalized`, `weighted_normalized`,
   `weighted_unnormalized`, `generalized` (GUniFrac, `alpha` param),
   plus an orthogonal `variance_adjust` flag (VAW) applicable to all
   5. skbio is missing `unweighted_unnormalized`, `generalized`, and
   `variance_adjust` entirely.

## Goal

One GPU-agnostic numba kernel, written once against the `numba.cuda`
API, that runs on NVIDIA (via `numba-cuda`) and AMD (via `numba.hip`'s
`pose_as_cuda()`) without maintaining two separate kernel
implementations, covering all 5 UniFrac methods plus
`variance_adjust`, in one PR. A numba CPU path exists as a secondary
fallback for users without a GPU, reusing the same formulas.

This is a deliberately large first PR (the user explicitly chose this
over a narrower per-method rollout), combining new GPU infrastructure
with new statistical methods. The risk of that combination is real and
accepted; mitigated by building and verifying method-by-method inside
the same PR rather than all at once at the end.

## Hard constraints

- **Pure Python only.** No new C++ build dependency for scikit-bio.
  `unifrac-binaries`/`ssu` remain reference/validation tools used
  during development only (the `unifrac-bench` conda env on scratch),
  never a runtime or test-suite dependency. `numba-cuda` and
  `numba.hip` (via `hip-python`) are new **optional** runtime
  dependencies (GPU extras), not required for base install.
- **Exactness.** SSU was chosen over DartUniFrac partly because SSU is
  exact where DartUniFrac is approximate. Default `fastmath=False` on
  both the GPU and CPU kernels, preserving exact IEEE-754 semantics.
  (SSU's own GPU build uses relaxed math; we choose not to, by
  default, for the same reason CPU stayed exact — this is a
  deliberate deviation from the reference's own build, not an
  oversight.)
- **GPU-agnostic by construction.** One kernel body, written against
  the `numba.cuda` API only. Backend selection happens at runtime
  (detect CUDA devices -> `numba-cuda`; else detect ROCm devices ->
  `numba.hip`'s `pose_as_cuda()`; else no GPU engine available), never
  by maintaining parallel CUDA-specific and HIP-specific kernel
  source.
- **Dual-backend verification is part of this PR**, not deferred:
  NVIDIA via NRP (reachable through the existing `kubectl`/Kubernetes
  setup) and AMD via native Cosmos `srun`. Both must actually run and
  match the checked-in `ssu` reference fixtures, not just one with the
  other "should work via `pose_as_cuda()`".

## Scope: the 5 methods and variance_adjust

Verified against the real `ssu` binary and SSU's source
(`unifrac_task_impl.hpp`, `unifrac-binaries` tag `v1.7`):

| SSU method | skbio today | This PR |
|---|---|---|
| `unweighted` | `unweighted_unifrac` (normalized only) | add `normalized=True` kwarg |
| `unweighted_unnormalized` | missing | via `unweighted_unifrac(normalized=False)` |
| `weighted_normalized` | `weighted_unifrac(normalized=True)` | kernel rewritten, same public behavior |
| `weighted_unnormalized` | `weighted_unifrac(normalized=False)` | kernel rewritten, same public behavior |
| `generalized` (alpha) | missing | new `generalized_unifrac(..., alpha=0.5)` function |
| `--vaw` (variance_adjust) | missing anywhere | `variance_adjust=False` kwarg on all three functions |

### Exact formulas (read directly from SSU source, not re-derived)

Let `p_u[k]`, `p_v[k]` be the proportional abundance of tree node `k`
(tip or internal, postorder) in samples u and v; `length[k]` its
branch length; `c_u[k]`, `c_v[k]` its raw counts; `T_u`, `T_v` the
samples' total tip counts.

- **`unweighted_unnormalized`**: `sum(length[k] for k where
  (p_u[k]>0) != (p_v[k]>0))`. Same numerator as the existing
  `_unweighted_unifrac`, never divided by the observed-branch-length
  denominator (`unifrac_task_impl.hpp`'s `UnnormalizedUnweightedTask`
  sets `compute_total = false`).
- **`generalized`** (`run_GeneralizedTask_T`,
  `unifrac_task_impl.hpp:925-983`, CPU branch): for each node where
  `sum = p_u[k] + p_v[k] != 0`:
  `numerator += length[k] * sum**(alpha-1) * abs(p_u[k]-p_v[k])`,
  `denominator += length[k] * sum**alpha`. Distance =
  `numerator/denominator`. Default `alpha=1.0` (matches `ssu --help`'s
  documented default).
- **`variance_adjust`**: for each node, `mi = c_u[k] + c_v[k]`, `m =
  T_u + T_v`, `vaw = sqrt(mi * (m - mi))`; nodes with `vaw <= 0`
  contribute nothing. Verified individually for every method (not just
  generalized):
  - `generalized` (`run_VawGeneralizedTask_T:1043-1119`): substitute
    `sum = (p_u[k]+p_v[k])/vaw`, `diff = abs(p_u[k]-p_v[k])/vaw` into
    the plain generalized formula above.
  - `weighted_normalized` (`run_VawNormalizedWeightedTask_T:849-922`):
    `numerator += length[k]*abs(p_u[k]-p_v[k])/vaw`, `denominator +=
    length[k]*(p_u[k]+p_v[k])/vaw`, distance = numerator/denominator.
  - `weighted_unnormalized` (`run_VawUnnormalizedWeightedTask_T:436-502`):
    same numerator, no denominator/division.
  - `unweighted` (`run_VawUnweightedTask_T:1756-1846`): `numerator +=
    length[k]/vaw` for nodes where presence differs, `denominator +=
    length[k]/vaw` for nodes where either sample is present, distance
    = numerator/denominator.
  - `unweighted_unnormalized` (`run_VawUnnormalizedUnweightedTask_T:1849+`):
    same numerator, no denominator/division.

  **Important:** skbio's existing plain `weighted_unifrac(normalized=True)`
  uses a tip-to-root-distance optimization (`node_to_root_distances`)
  that exploits linearity to avoid summing over internal nodes. This
  does **not** generalize to `generalized_unifrac` or any
  `variance_adjust` variant, since the alpha-power and
  `sqrt(mi*(m-mi))` terms are nonlinear. All new kernels sum directly
  over every node (tips + internal) using plain `branch_lengths` and
  the per-node arrays `_setup_multiple_unifrac` already produces; only
  the existing, untouched plain `weighted_normalized` path keeps using
  `node_to_root_distances`.

`generalized_unifrac` and `variance_adjust=True` are **numba-only**,
no cython fallback: raise `ImportError` when numba isn't installed,
matching the existing pattern in commit `94b40ae5`. The existing
`unweighted_unifrac`/`weighted_unifrac` cython path is untouched.

## Architecture

### Kernel design

The GPU kernel directly ports SSU's actual GPU loop shape (confirmed
clean in the earlier feasibility research): a `(sample_block, stripe,
sample_in_block)` grid, one output cell per thread, no atomics, no
warp-level tricks, persistent device-resident buffers allocated once
and reused across the whole tree traversal (`cuda.device_array`/
`copy_to_device` once, matching SSU's `acc_create_buf`/
`acc_update_device` lifecycle). Each thread accumulates its pair's
distance by looping serially over a batch of tree nodes ("embeddings")
per kernel launch, same as SSU, to bound per-launch memory.

One kernel body is parameterized by an integer method enum
(`UNWEIGHTED`, `UNWEIGHTED_UNNORMALIZED`, `WEIGHTED_NORMALIZED`,
`WEIGHTED_UNNORMALIZED`, `GENERALIZED`) plus `alpha` and
`variance_adjust`, written once against `numba.cuda`. Backend
selection is a thin runtime dispatch layer, not a second kernel:

```
detect_gpu_backend() -> "cuda" | "hip" | None
    - "cuda": CUDA device present -> import numba.cuda directly (numba-cuda package)
    - "hip": no CUDA device, ROCm device present -> import numba.hip,
      call numba.hip.pose_as_cuda(), then the SAME kernel source runs
      unmodified against the HIP backend
    - None: no GPU -> fall back to the CPU numba path
```

Tree traversal and proportion-propagation preprocessing
(`_setup_multiple_weighted_unifrac`-style functions) is reused as-is
for both CPU and GPU paths; profiling confirmed it's negligible cost
and there is no correctness reason to replace it.

The CPU fallback path extends the **existing** row+mirror kernel
pattern (`_unweighted_unifrac_pdist_nb`, `_weighted_unifrac_pdist_nb`)
with the same formulas, rather than reusing the GPU kernel's stripe
loop — the CPU and GPU kernels are two separate implementations of the
same formulas, not one shared code path, because `numba.cuda` and
`@njit(parallel=True)` are different idioms with no meaningful code
sharing between them.

### Output

Output stays condensed (vector form) throughout on both CPU and GPU
paths, passed to `DistanceMatrix(condensed, ids)` directly (already
supported, `skbio/stats/distance/_base.py:205-218`, no `squareform()`
call needed). This keeps Phase-1-style forward compatibility with
Qiyun's (not yet implemented) distance-matrix refactor plan: its
planned `vector` storage flag and `binary_dm` lazy-access HDF5
extension are the tracked, separate answer to large-N output-matrix
memory growth; this work only needs to stay condensed-friendly, not
implement disk-backed output itself.

### Public API

- `unweighted_unifrac(..., normalized=True, variance_adjust=False)`
- `weighted_unifrac(..., normalized=False, variance_adjust=False)`
  (unchanged signature, `variance_adjust` added)
- `generalized_unifrac(u_counts, v_counts, taxa, tree, alpha=1.0,
  variance_adjust=False, validate=True)` (new)
- All three wired into `beta_diversity()`'s metric registry.
- Engine selection: existing `engine={'cython','numba','fast'}` is
  unchanged for `unweighted_unifrac`/`weighted_unifrac`'s existing
  behavior. A new engine value is added for GPU dispatch (exact name
  TBD during implementation, e.g. `engine='gpu'`), vendor-agnostic —
  the caller never specifies CUDA vs. HIP, the runtime backend
  detection above handles that.

## Error handling

- `alpha` for `generalized_unifrac` must be in `[0, 1]`; raise
  `ValueError` outside that range.
- `generalized_unifrac` and `variance_adjust=True` raise `ImportError`
  with an actionable message when numba is not installed.
- Requesting the GPU engine with no GPU backend detected raises a
  clear error (not a silent CPU fallback) when the engine was
  explicitly requested; falls back silently to CPU only when the
  caller asked scikit-bio to choose automatically (mirrors the
  existing `engine='numba'` vs `engine='fast'` distinction in
  `_numba_unifrac_fast_path_eligible`).
- Existing `validate=True` taxa/tree checks are reused unchanged.
- Degenerate inputs (both-samples-empty) get test coverage extended
  from the existing pattern (commit `a0136c9c`) to the new
  methods/kwargs and to both CPU and GPU paths.

## Testing

Extend `skbio/diversity/beta/tests/data/qiime-191-tt/` (the existing
QIIME 1.9.1 tiny-test fixture, currently holding reference output for
3 of the methods, generated by an external tool, never a test-time
dependency) with new checked-in `ssu`-generated reference outputs for
`unweighted_unnormalized`, `weighted_unnormalized`, `generalized`
(alpha=0.5 and alpha=1.0), and variance-adjusted variants of all 5
methods. `unifrac`/`ssu` is a development-time tool only, never
installed in CI.

**Dual-backend verification, both required, not optional for this
PR:**
- NVIDIA: run via NRP (`kubectl`-submitted job), compare against the
  `ssu` fixtures.
- AMD: run natively on Cosmos (`srun`, MI300A / gfx942), compare
  against the same fixtures.
- Both backends must independently match the fixtures within a
  measured tolerance (not assumed; measured once implemented and
  documented with the actual figure, same discipline as commit
  `29f6ac42`'s message) — this is the actual proof that "one kernel,
  two backends" holds, not just that `pose_as_cuda()` runs without
  erroring.

CPU-path testing: same fixture comparisons, plus the existing
both-samples-empty/single-tip edge cases extended to the new
methods/kwargs, plus `ImportError`/`ValueError` dispatch tests, plus
`beta_diversity()` wiring tests.

## Out of scope for this spec

- Reviving the unused existing `qiime-191-tt` beta fixtures as a
  cleanup (flagged, not fixed here).
- Any change to `skbio/io/format/binary_dm.py` or the distance-matrix
  refactor plan.
- A narrower first-PR scope (e.g. `weighted_normalized`-only to prove
  out infra before adding methods) was proposed and explicitly
  declined in favor of all 5 methods + variance_adjust together.
