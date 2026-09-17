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

## Install

```bash
pip install -r requirements.txt
```

## Reproduce the benchmarks

```bash
python benchmark.py            # standalone codec: scaling + variance + ZFP matched-error
python compare_blocked.py      # tiered vs block-local vs ZFP
python conservation_test.py    # downstream conservation drift
```

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
