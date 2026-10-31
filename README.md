# Conserved-Field Compression

**Lossy compression that keeps the physics exact.**

> **Paper:** *Codec-Agnostic Exact Conservation in Lossy Scientific Data Compression* - Submitted to IWBDR-6, the 6th International Workshop on Big Data Reduction at IEEE BigData 2026.

Lossy compressors keep every value within an error bound, but totals like mass and energy quietly drift, and the drift grows the harder you compress. This project adds a closed-form layer that makes those totals **exact** after decompression, for **two stored numbers per field**, with any compressor and without changing its pointwise error bound.

- **Up to 9 orders of magnitude less drift.** On NYX cosmology data compressed 1,304× with SZ3, energy drift drops from 3.1×10⁻³ to 2.3×10⁻¹².
- **Works with any codec.** Validated with ZFP and SZ3 on CPU, and with PFPL and SLEEK on GPU.
- **Real data.** SDRBench climate (CESM), weather (Hurricane ISABEL) and cosmology (NYX) fields, from 1D line-outs to full 3D.
- **Nearly free.** 16 bytes per field. The ratio cost is 0–15% from a 10% bound tightening, and often zero.
- **Fast.** With SLEEK on an RTX 4090, compression runs at about 350 GB/s and decompression at about 190 GB/s with mass and energy both conserved.
- **Proven.** Exactness and error-bound guarantees are derived in closed form in the paper.

| Data | Codec | Ratio | Energy drift: codec → with layer | Mass drift: codec → with layer |
|---|---|---:|---|---|
| NYX velocity | SZ3 | 1304× | 3.1e-3 → **2.3e-12** | 4.5e-5 → **4.0e-12** |
| ISABEL temperature | SZ3 | 903× | 1.5e-3 → **1.2e-11** | 6.7e-4 → **4.0e-12** |
| CESM humidity | PFPL (GPU) | 56× | 1.4e-4 → **1.6e-10** | 7.0e-4 → **1.2e-10** |
| NYX velocity | ZFP | 22.5× | 1.7e-5 → **3.2e-11** | 3.7e-6 → **7.3e-11** |

## How it works

Before compression, store the field's total mass M₀ = Σxᵢ and energy S₀ = Σxᵢ². After decompression, one affine map restores both exactly. An offset fixes the mean, and a scale about the mean fixes the variance; neither disturbs the other:

```
C(x̂ᵢ) = mean₀ + a·(x̂ᵢ − mean),   a = √(var₀ / var)
```

Mass only, energy only, weighted invariants such as momentum, and mass+energy together are all supported through one interface.

## Get started

The current implementation is in [`corrector/`](corrector/):

| File | What it is |
|---|---|
| `conserve.h` | CPU reference (C/C++, optional OpenMP) |
| `conserve_cuda.h` | GPU version (CUDA) |
| `cSLEEK/`, `cPFPL/` | Integrations with the SLEEK and PFPL GPU compressors |
| `sz3_adapter.py`, `rd_full.py` | Python benchmark harness for SZ3 and ZFP |
| `results/RESULTS.md` | GPU results on an RTX 4090 |

```c
#include "conserve.h"
// before compression: M0 = sum(x), S0 = sum(x*x) over the original field
// after decompression, in place:
conserve_mass_energy(xhat, N, M0, S0);   // mass and energy now exact
```

Ideas, questions and collaborations are welcome. Open an issue or reach out.

---

## Where it started: ZFP+ and the standalone codec
---

## Dive in: ZFP+ (`zfp_plus.py`)

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

An attention-routed, block-local quantizer initially mapped the space.
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

## What did NOT work


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

3D + a field-local stabilizer (KPBM) — the boundary, pushed back. The 3D per-node restore alone only delays divergence (above). Pairing it with a shear-gated dissipation term (KPBM) closes that gap: at ZFP tol 3e-4, where plain BGK compressed-in-loop crashes at step 2106, the stabilized run survives the full 2500 steps and tracks the true (uncompressed) flow to ~3.5%. Sphere at Re=200, D3Q19, error measured against the α=0 uncompressed reference:

config	result	err vs. true flow
BGK (α=0), compressed	crash @2106	—
KPBM α=0.10	stable	3.50%
KPBM α=0.25	stable	3.47%
KPBM α=0.50	stable	3.74%
uniform viscosity (matched avg.)	stable	4.46%

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
