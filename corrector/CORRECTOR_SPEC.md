---
title: "Conservation Corrector — Integration Spec (for SLEEK / PFPL)"
author: "Alika M. Parks — for M. Burtscher"
date: "October 2026"
geometry: margin=1in
fontsize: 11pt
---

Reference code: `conserve.h` (public-domain reference; CPU reference + CUDA mapping
in comments). This spec is what you need to wire it into SLEEK / PFPL. Scope is the
**absolute error bound (ABS)**; the relative-bound variant is noted at the end.

## What it does

A lossy codec preserves a pointwise error bound but not integrated invariants
(mass, energy, momentum). The corrector restores chosen invariants **exactly**, in
closed form, from two stored scalars, touching only the decompressed array — so it is
codec-agnostic and drops in after decode. The whole correction is a single affine map
`x ← a·x + b`, so **two scalars `(a,b)` cover every case** (mass, energy, momentum,
mass+energy). Verified on a reference field: mass and energy drift 7.5e-5 → 3e-8
(float32 floor).

## Three integration points

1. **At compression — compute and store `(a,b)`.** Because PFPL (and SLEEK's
   bound-checked path) already decompresses the data during compression to verify the
   error bound, you have the reconstructed field `x̂` in hand at encode time. Compute the
   correction's two scalars there and store them as side-info in place of the raw
   invariants:
   - mass+energy: `conserve_mass_energy_params(x̂, N, M0, S0, &a, &b)`
   - mass (or a momentum component): `conserve_linear_params(x̂, N, L0, w, &a, &b)` → `a = 1`
   - energy only: `conserve_energy_params(x̂, N, S0, &a, &b)` → `b = 0`

   `M0=Σx`, `S0=Σx²`, `L0=Σ(w·x)` are taken from the **original** field. Only `(a,b)`
   are stored — two doubles per field/block, the only bytes the layer adds. (Fallback
   for codecs that do *not* decompress at encode: store `(M0,S0)` instead and let the
   decoder recover `(a,b)` with one reduction pass — see point 3.)

2. **Compress at a slightly tightened error bound.** The correction moves each value by
   a known displacement `Δ`, so run the codec at a tightened bound `b' = (1−ε)·b` to keep
   the *total* error inside the user's requested bound `b`. This holds for **any** `b`
   (uniform tightening — no special-casing of rounding buckets).
   - **mass / momentum (offset):** the displacement is the uniform offset, `|Δ| = |b|
     ≤ b'`, so a small `ε` suffices and `b' + |b| ≤ b` is checkable at encode.
   - **mass+energy (affine):** the displacement `(a−1)(x̂ᵢ−mean) + β₀` grows with distance
     from the mean, so a flat `ε` is **not** sufficient. Use
     `conserve_mass_energy_maxdisp(x̂, N, M0, S0)` to get `maxᵢ|Δᵢ|`, then either tighten
     by it or — since you already have `x̂` and the original — apply `(a,b)`, check the
     corrected field against `b`, and retighten on the rare miss. Either gives a hard
     guarantee. Report the small ratio cost of the tightening.

3. **After decompression — correct in place, one pass.** With `(a,b)` stored:
   - `conserve_apply_affine(x, N, a, b)` → `x[i] = a·x[i] + b`.

   That's a single elementwise pass, **no reduction** on the decode side. (Fallback
   path, if you stored `(M0,S0)` instead: `conserve_mass_energy(x, N, M0, S0)` /
   `conserve_linear(x, N, L0)`, which do the reduction at decode.)

## Modes: standalone CLI vs. API

- **CLI (standalone SLEEK/PFPL):** expose **mass** and **energy** — each is single-field
  with a scalar target, so the user just names the invariant (like the error bound).
- **API / library:** **momentum** lives here. Its weight `w` is the **density field**
  (momentum = `Σ ρᵢvᵢ`), i.e. another field in the dataset — not values the user types.
  The caller passes the density field pointer as `w`; the encoder uses it to compute the
  single scalar `b`, and the **decoder needs no vector at all** (it just applies `(1,b)`).
  If density is itself corrected (mass), correct it first and use the corrected `ρ̂` as `w`.

## The method (why it is exact and always real)

For mass+energy the correction is a **decoupled affine**: an offset sets the mean
(fixing mass) and a scale about the mean sets the variance (fixing energy). Scaling
about the mean leaves the sum unchanged and the offset leaves the variance unchanged,
so the two never interfere — exact and always real (no root-selection, unlike a raw
`a·x+b` match). Collapsing to `a·x̂+b` with `b = mean0 − a·mean` reproduces it exactly
in one pass. Full derivation in `conserve.h`.

**Taxonomy:** linear invariant → offset; quadratic invariant → scale-about-mean;
invariants on separate fields are independent; `k` invariants on one field → a small
`k`-parameter solve (the 2×2 energy+momentum case is written out in `conserve.h`).

## GPU mapping (per field/block)

- **Encode:** the reductions `Σx̂`, `Σx̂²` (fused kernel or `cub::DeviceReduce`, double or
  Kahan) happen in the decompress you already run for bound-checking; compute `(a,b)` on
  the host (2–3 scalars) and store them.
- **Decode:** one elementwise FMA kernel, `x[i] = a·x[i] + b` — **no device-wide
  reduction, no host round-trip.** Fuses naturally with SLEEK's own decode sweep.

Cost is O(N) traffic, negligible next to the codec.

## What we need from the integration (for the paper)

- Real coded **ratios** (SLEEK/PFPL full pipeline), layer on/off.
- **Throughput** with and without the corrector, on your GPU system(s).
- Invariant drift, layer on/off, confirming the ~machine-precision restore holds in the
  GPU path as it does on CPU.

## Verification checklist

- [ ] After correction, `|Σx − M0|/|M0|` and `|Σx² − S0|/|S0|` at the float floor (~1e-7–1e-8 in fp32).
- [ ] Max pointwise error stays within the user's bound `b` (offset: via `ε`; mass+energy: via `maxdisp`/verify).
- [ ] Single-pass `conserve_apply_affine(a,b)` matches the two-pass reference bit-for-bit (up to fp rounding).
- [ ] Throughput overhead is a small fraction of decode time.
