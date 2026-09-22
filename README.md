# Conserved-Field Compression

**Make lossy scientific compression conserve the physics.**

A closed-form conservation layer for ZFP and other lossy scientific codecs.
Exact energy conservation, near-zero compression-ratio overhead, and no change
to the codec's pointwise error bound. (Energy today; the same closed-form
method extends to other invariants.)

Lossy compressors for scientific floating-point grids (ZFP, SZ, TTHRESH)
guarantee a *pointwise* error bound. They do **not** guarantee that
*integrated physical invariants* — total energy, mass, momentum — survive
compression. Downstream analysis that checks an energy budget or a
conservation law sees the compressed field drift, and the drift grows the
harder you compress.

This project adds an exact conservation guarantee on top of an existing lossy
codec, for a handful of bytes, in closed form. It also includes a standalone
attention-routed block codec used to develop the idea.

> **Status: research code.** Validated on synthetic 3D turbulence. Not yet
> tested on production simulation output. Numbers below are reproducible from
> this repo (`colab.ipynb` → Run all, or the scripts in "Reproduce").

---

## The headline: ZFP+ (`zfp_plus.py`)

Compress with ZFP, then solve in closed form for the single scale factor
that makes the reconstruction's total energy equal the original's, and store
it (4 bytes/component). ZFP's ratio and pointwise error are preserved; total
energy becomes exact.

64³ synthetic turbulence, mean over 3 seeds:

| ZFP tolerance | ratio | max abs error | ZFP energy drift | **ZFP+ energy drift** |
|---:|---:|---:|---:|---:|
| 0.02 | 14.1× | 0.0025 | 5.5e-6 | **3.4e-10** |
| 0.1  | 22.2× | 0.0093 | 2.5e-5 | **2.9e-10** |
| 0.3  | 35.9× | 0.033  | 7.1e-5 | **1.3e-10** |
| 1.0  | 59.7× | 0.109  | 4.8e-4 | **7.1e-9**  |

The correction is invisible in the compression ratio and pointwise error; it
drives energy drift down by 3–4 orders of magnitude, and the benefit grows
with compression aggressiveness — exactly the regime where conservation
matters. The same wrapper works on any lossy backend, not just ZFP.

```python
import numpy as np, zfp_plus
field = np.random.rand(64, 64, 64).astype(np.float32)
payload, recon = zfp_plus.compress(field, tolerance=0.1)   # energy-conserving
restored = zfp_plus.decompress(payload)                     # == recon
```

---

## The standalone codec (`blocked.py`, `attention.py`, …)

An attention-routed, block-local quantizer developed while exploring the idea.
A physics-informed attention map (vorticity magnitude) routes per-block
bit-precision; each block quantizes against its own local min/max; an
algebraic layer enforces exact energy conservation.

The one change that mattered was **block-local quantization** (per-block
min/max instead of one global range per tier): it took the standalone codec
from 2.97× to **5.56×** while cutting max error ~6×, with energy conserved to
~1e-8.

ZFP+ outperforms this standalone codec on every axis (ratio, error,
conservation), which is *why* the recommended path is the conservation layer
on top of ZFP rather than a bespoke codec. The standalone codec is retained
for reference and for the attention/constraint machinery.

```python
import numpy as np
from attention import vorticity_attention_map
from blocked import quantize_blocked, dequantize_blocked
u = np.random.rand(64,64,64).astype(np.float32)
att = vorticity_attention_map(u, u, u, smooth_sigma=0.6)
bf = quantize_blocked(u, att, block_size=8)
recon = dequantize_blocked(bf)   # energy-conserving, block-local
```

---

## What did NOT work (documented dead-ends)

Honest negative results, kept behind flags so they can be retested on real
(anisotropic) data where they may behave differently:

- **Downsampling smooth blocks** (`downsample="const"|"tri"`, `blocked.py`):
  even trilinear-upsampled, buys ~+0.1× ratio for a large error increase. The
  background is already at 4 bits, so there's little to save.
- **Auto tier thresholds from attention quantiles** (`auto_tiers`,
  `blocked.py`): fixed percentile cuts came out worse than hand-picked
  thresholds. Would need optimization against a rate/error objective.
- **Component-frame rotation** (`rotated.py`): per-block PCA concentrates 91%
  of energy into one component, exactly and reversibly — but it does **not**
  help compression, because block-local quantization is scale-invariant
  (it already normalizes each component to its local range), so concentrating
  magnitude buys nothing, while per-block rotation adds block-seam
  discontinuities and quaternion overhead. A clean insight: *once you quantize
  against local ranges, an energy-concentrating transform stops helping.*

---

## Beyond storage: in-loop use

Because the layer only ever touches decompressed arrays, the same closed-form
correction can sit *inside* a memory-bandwidth-bound solver, not just on its
output files. Lattice-Boltzmann (LBM) is the motivating case: it is limited by
the memory traffic of streaming its distribution functions, so compressing them
in-loop could free bandwidth and fit larger domains on a single node — *if the
solver survives the per-step error.* Uncorrected, it does not: truncating the
distributions each step injects a biased error that drives the solver unstable.
Restoring each node's conserved moments (mass and momentum) after compression is
what keeps it alive. Measured results below; both are reproducible.

**2D cylinder flow — the correction works.** D2Q9, Re=10 (a stable BGK
baseline), truncation proxy at step `q`, 2000 steps. Uncorrected in-loop
compression blows up at every level tested; the per-node moment restore prevents
it at all levels. The accuracy retained scales with how hard you compress:

| `q` (truncation step) | uncorrected | corrected | corrected field-L2 err |
|---:|:--|:--|---:|
| 5e-4 | blows up @750 | **stable** | 3.1% |
| 1e-3 | blows up @200 | **stable** | 5.2% |
| 2e-3 | blows up @100 | **stable** | 9.2% |
| 4e-3 | blows up @100 | **stable** | 18.5% |

The correction converts divergence into a stable run; at modest compression that
run tracks the true (uncompressed) solution to ~3%, degrading as compression
gets aggressive. The base solver must itself be stable for this to mean
anything — BGK below tau≈0.55 is unstable on its own, independent of any
compression, and such a regime is *not* a valid test of the correction.

**3D flow past a sphere — a characterized limit.** D3Q19, Re=200, real ZFP
backend. Here the same per-node restore is *not* sufficient. It improves fidelity
and yields a net 7–11× compression at tight tolerance, but at looser tolerances
both corrected and uncorrected runs diverge — the correction only delays the
crash:

| ZFP tol | uncorrected | corrected | net ratio (corrected) |
|---:|:--|:--|---:|
| 1e-4 | stable, 0.92% | stable, **0.70%** | 7.4× |
| 3e-4 | crash @1710 | crash @2106 | 7.9× |
| 1e-3 | crash @763 | crash @883 | 9.1× |
| 3e-3 | crash @358 | crash @376 | 11.3× |

The boundary is informative: restoring the *conserved* moments (0th, 1st)
controls 2D stability but not 3D, where the momentum-flux and higher ("ghost")
moments — which compression also corrupts and the restore leaves untouched —
drive the instability. Stabilizing 3D in-loop compression would require
constraining those higher moments (regularized-LBM territory), a separate and
larger method than a conservation layer.

**1D wave equation — exact joint conservation for storage (`wave1d.py`).**
When a field and its time-derivative `(u, u_t)` are compressed as one record,
total energy splits into `KE = ½Σv²` and `PE = ½c²Σ(∂u/∂x)²` — two quadratic
invariants living on different parts of the record. A three-scalar affine
correction restores KE, PE, total energy, *and* mass to machine precision
(≤2e-16), each scalar acting on its own invariant: `a_v` scales `v` (fixes KE),
`a_u` scales `u` (fixes PE), and an offset `b` on `u` restores mass without
touching PE, since a constant leaves every spatial difference unchanged. This is
the "conserve multiple invariants at once" roadmap item, demonstrated exactly.
(Applied in-loop to a non-dissipative leapfrog it only delays a blow-up — a
global scalar constraint, unlike the per-node restore, does not control the
per-mode error.)

Paired with a workstation-scale stabilizer such as KPBM
(github.com/alikamp/kpbm-workspace), the storage and 2D results point toward
*usable transient/quasi-steady 2D CFD with in-loop compression on commodity
hardware*; the 3D limit is characterized above rather than glossed.

## Paper

A short methods draft is in [`paper/`](paper/):
[PDF](paper/conserved_field_compression_paper.pdf) ·
[LaTeX source](paper/conserved_field_compression_paper.tex). Draft, not yet
submitted.

## Install

```bash
pip install -r requirements.txt
```

## Reproduce the benchmarks

```bash
python benchmark.py            # standalone codec: scaling + variance + ZFP matched-error
python compare_blocked.py      # tiered vs block-local vs ZFP
python conservation_test.py    # downstream conservation drift
python wave1d.py               # 1D wave: exact joint (KE, PE, mass) restoration
python kpbm_inloop.py          # 2D LBM in-loop: uncorrected diverges, corrected stays stable
```

The 2D in-loop numbers in "Beyond storage" are at Re=10 (a stable BGK baseline);
`kpbm_inloop.py` defaults must keep the base solver stable (tau ≳ 0.55) for the
comparison to be meaningful. The 3D sphere numbers come from a separate D3Q19
run (github.com/alikamp/kpbm-workspace).

Or open `colab.ipynb` in Google Colab and **Runtime → Run all** (installs its
own dependencies).

## Roadmap

- Conserve **multiple** invariants at once (energy + mass → a 2-parameter
  affine solve; energy + momentum → constrained least squares).
- **Attention-gated** correction: confine the rescale to smooth regions to
  protect ZFP's pointwise bound inside features.
- Wrap **SZ** as an additional backend; compare on SDRBench.
- Validate on real CFD / cosmology snapshots.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Author

Alika Parks · [github.com/alikamp](https://github.com/alikamp)
