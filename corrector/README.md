# Conservation corrector — reference implementation

A codec-agnostic, closed-form layer that restores integral invariants (mass,
energy, momentum) **exactly** after lossy compression, from a few stored
scalars, without touching the codec or its pointwise error bound. This is the
reference drop for integration into SLEEK / PFPL (or any lossy codec) and the
backbone harness for the IWBDR-6 storage-case study.

## Files

| File | What it is |
|------|------------|
| `conserve.h` | Header-only reference corrector. **ABS** paths: `conserve_mass_energy` (decoupled affine), `conserve_linear` / `conserve_linear_w` (mass / weighted momentum), `conserve_apply(x, N, which, M0, S0, w)`. **REL** paths: `conserve_scale_mass` and `conserve_apply_rel` (multiplicative, bound-preserving under a relative bound). `conserve_mass_energy_maxdisp` for the ABS verify/tighten protocol. CPU reference + CUDA mapping in comments. |
| `conserve_test.cpp` | Self-contained correctness test for `conserve.h` (invariants restored to ~1e-10 relative; pointwise bound preserved). |
| `CORRECTOR_SPEC.md` / `.pdf` | Integration spec: the three integration points, uniform bound-tightening `b' = (1-eps)*b`, the GPU mapping (two reductions + one fused multiply-add), and a verification checklist. |
| `rd_full.py` | Systematic storage-case rate-distortion harness (E1/E2): real coded ratios per codec with mass/energy drift layer-on/off, pointwise-bound check, PSNR. ZFP works anywhere; SZ3 / MGARD / ZFP-ROUND_FIRST via libpressio. |
| `rd_full_colab.txt` | Colab runner for the backbone sweep on real SDRBench data (ZFP immediately; libpressio cells for the rest). |
| `run_sdrbench_colab.txt` | Fetches the SDRBench Hurricane ISABEL fields on Colab. |
| `fig_backbone.py` | Regenerates the dimensionality figure (ZFP on ISABEL, 1D/2D/3D, raw vs corrected). |

## Quick start

```sh
# C++ reference correctness
g++ -O2 conserve_test.cpp -o ctest && ./ctest

# Python RD harness, ZFP self-test (synthetic 3D + 2D slice + 1D line-out)
pip install zfpy numpy
python rd_full.py

# Real data: run rd_full.py field.npy on an SDRBench field, or use the Colab runner.
```

## Invariant selection

*Which* invariant to preserve is a user option (like the error bound), set per
job, not a property of the data:

```c
#include "conserve.h"
/* --- absolute bound (ABS) --- */
/* mass + energy together (decoupled affine, always real) */
conserve_apply(x, N, CONS_MASS | CONS_ENERGY, M0, S0, NULL);
/* momentum of a velocity field v: weight w = density (sum rho*v) */
conserve_apply(x, N, CONS_MASS, L0, 0.0, rho);

/* --- relative bound (REL): multiplicative, single invariant --- */
conserve_apply_rel(x, N, CONS_MASS,   M0, S0);   /* scale alpha = M0/sum(x) */
conserve_apply_rel(x, N, CONS_ENERGY, M0, S0);   /* scale a = sqrt(S0/sum x^2) */
```

The correction must match the codec's bound mode: ABS corrections are
additive/affine; REL corrections are multiplicative (an additive offset is
unbounded as a value approaches zero). Joint mass+energy under REL has no
single global affine and `conserve_apply_rel` returns 0 for it. For the ABS
mass+energy path, `conserve_mass_energy_maxdisp` gives the largest displacement
the correction adds, so the encoder can tighten the bound (or verify and
recompress) to keep the pointwise guarantee. See `conserve.h` for the
derivation.

Store the targets (`M0 = sum(x)`, `S0 = sum(x*x)`, weighted `L0 = sum(w*x)`)
on the **original** field before compression — a handful of bytes per field.
