/*
  conserve.h  —  Reference conservation corrector for lossy scientific codecs.
  A. M. Parks, 2026.  Public reference implementation for integration into
  SLEEK / PFPL (or any lossy codec).  Post-decompression, in place.

  Idea: a lossy codec preserves a pointwise error bound but not integrated
  physical invariants (mass, energy, momentum). This layer restores chosen
  invariants exactly, in closed form, from a few stored scalars, without
  changing the codec. It touches only the decompressed array, so it is
  codec-agnostic and drops in after decode.

  Integration points (per field, or per block):
    (1) BEFORE compression: compute the target invariants on the ORIGINAL
        field and store them as side-info (a few bytes; see below).
    (2) Compress at a slightly TIGHTENED error bound so the corrector's
        rescale stays inside the user's requested bound (see SPEC).
    (3) AFTER decompression: call the matching conserve_* routine in place.

  All reductions are in double; the array stays in the codec's precision.
  GPU mapping (per routine): parallel reduction(s) to get the sums, compute
  2–3 scalars on the host, then one elementwise (fused multiply-add) kernel.
*/
#ifndef CONSERVE_H
#define CONSERVE_H
#include <math.h>

/* ---------------------------------------------------------------------------
   Mass + energy (the workhorse).  Decoupled affine, ALWAYS real.
     targets (stored, computed on the original):
        M0 = sum(x)         (mass,   a linear invariant)
        S0 = sum(x*x)       (energy, a quadratic invariant)
     restore:
        mean0 = M0/N ;  var0 = S0/N - mean0*mean0        (>= 0)
        mean  = sum(x)/N ;  var = sum(x*x)/N - mean*mean
        a     = sqrt(var0/var)                           (var>0, else 1)
        x[i] <- mean0 + a*(x[i] - mean)
   Scaling about the mean leaves the sum unchanged (so setting the mean to
   mean0 fixes mass), and the offset leaves the variance unchanged (so the
   scale fixes energy). The two never fight -> exact, no root selection.
   Side-info: M0, S0  (two scalars).
--------------------------------------------------------------------------- */
static inline void conserve_mass_energy(float* x, long N, double M0, double S0)
{
    double s = 0.0, s2 = 0.0;
    for (long i = 0; i < N; i++) { double v = x[i]; s += v; s2 += v * v; }   /* reduce */
    const double mean  = s  / (double)N;
    const double var   = s2 / (double)N - mean * mean;
    const double mean0 = M0 / (double)N;
    const double var0  = S0 / (double)N - mean0 * mean0;
    const double a = (var > 0.0) ? sqrt(var0 / var) : 1.0;
    for (long i = 0; i < N; i++)                                             /* map   */
        x[i] = (float)(mean0 + a * ((double)x[i] - mean));
}

/* ---------------------------------------------------------------------------
   Any single LINEAR invariant  L0 = sum(w*x)  (mass: w=1; momentum on a
   component: w = density, or w=1 for a velocity-sum convention).  Offset only.
   For w=1 this is mass; pass a weight array for a weighted linear invariant.
   Side-info: L0 (one scalar).  Offset does not perturb any energy term that
   depends on differences of x, so it composes with the energy scale above.
--------------------------------------------------------------------------- */
static inline void conserve_linear(float* x, long N, double L0)
{
    double s = 0.0;
    for (long i = 0; i < N; i++) s += x[i];
    const double b = (L0 - s) / (double)N;
    for (long i = 0; i < N; i++) x[i] = (float)((double)x[i] + b);
}

/* ---------------------------------------------------------------------------
   WEIGHTED linear invariant  L0 = sum(w_i * x_i), with a per-element weight
   field w (e.g. momentum of a velocity field v is sum(rho_i * v_i), so w=rho).
   The offset that preserves it:
        want sum(w*(x+b)) = L0  ->  b = (L0 - sum(w*x)) / sum(w)
   w == NULL falls back to the unweighted case (w=1). The weight is the user's
   choice and is representation-dependent: if the stored field is already a
   momentum density (rho*v) then w=1; if it is velocity with density in a
   separate field, pass that density as w.
--------------------------------------------------------------------------- */
static inline void conserve_linear_w(float* x, long N, double L0, const float* w)
{
    if (!w) { conserve_linear(x, N, L0); return; }
    double swx = 0.0, sw = 0.0;
    for (long i = 0; i < N; i++) { swx += (double)w[i] * x[i]; sw += (double)w[i]; }
    const double b = (sw != 0.0) ? (L0 - swx) / sw : 0.0;
    for (long i = 0; i < N; i++) x[i] = (float)((double)x[i] + b);
}

/* ---------------------------------------------------------------------------
   WHICH invariant to preserve is a USER choice, not a property of the input.
   The same field may be compressed preserving mass for one downstream analysis
   and energy for another — it depends on what the consumer checks, not on the
   data. So expose it like the error bound: one compressor option the caller
   sets per job. ONE dispatch handles every case; no need for separate builds.

     which (bitmask): CONS_MASS | CONS_ENERGY   (momentum = CONS_MASS on a
                      velocity component field; vector momentum = apply per
                      component). Pass the targets named by `which`.
--------------------------------------------------------------------------- */
enum { CONS_MASS = 1, CONS_ENERGY = 2 };

/* which: bitmask of invariants to preserve (user option, set per job).
   M0,S0: the stored targets named by `which`.
   w:     optional weight field for the linear invariant (momentum: w=density;
          NULL = unweighted, i.e. mass). Ignored for the energy-only path. */
static inline void conserve_apply(float* x, long N, int which,
                                  double M0, double S0, const float* w)
{
    const int want_m = (which & CONS_MASS)   != 0;
    const int want_e = (which & CONS_ENERGY) != 0;
    if (want_m && want_e) { conserve_mass_energy(x, N, M0, S0); return; }  /* offset + scale */
    if (want_m)           { conserve_linear_w(x, N, M0, w);     return; }  /* mass/momentum: (weighted) offset */
    if (want_e) {                                                          /* energy only: global scale */
        double s2 = 0.0; for (long i = 0; i < N; i++) { double v = x[i]; s2 += v*v; }
        const double a = (s2 > 0.0) ? sqrt(S0 / s2) : 1.0;
        for (long i = 0; i < N; i++) x[i] = (float)(a * (double)x[i]);
    }
}

/* ===========================================================================
   BOUND MODES.  The correction must match the codec's pointwise bound mode,
   or it can violate the very bound the codec guaranteed.

   ABS  (|x - xhat| <= eps):
     The corrections above are additive/affine and preserve ABS *with headroom*.
       - conserve_linear: adds a uniform offset beta = (M0 - sum xhat)/N, which
         is the mean of the signed codec errors, so |beta| <= b'. The corrected
         error is <= b' + |beta|. Compress at b' = (1-e)*eps and the encoder can
         check b' + |beta| <= eps (beta is known at encode) -> hard guarantee.
       - conserve_mass_energy: affine C_i = a*xhat_i + beta0, so the displacement
         from the true value is (a-1)*(xhat_i - mean) + offset; it is LARGEST for
         points far from the mean and a flat tightening does NOT cover it. Use
         conserve_mass_energy_maxdisp() to get max|Delta| and tighten by it, or
         (since Delta is deterministic) verify the corrected field against eps at
         encode and recompress on the rare miss.

   REL  (|1 - xhat/x| <= eps, same sign, x != 0):
     An ADDITIVE offset is unbounded as x -> 0 (a fixed shift vs. a shrinking
     allowance), so conserve_linear must NOT be used under REL. The bound is
     multiplicative, so the corrections must be too:
       - mass   -> global scale alpha = L0 / sum(xhat)   (conserve_scale_mass)
       - energy -> global scale a     = sqrt(S0 / sum xhat^2)
     Each scales every value equally, so it preserves REL (the bound inflates
     only by |scale - 1|). Joint mass+energy has no single global affine that
     stays within REL near zero, so apply one invariant, or a per-element
     multiplier out of band (see conserve_apply_rel).
--------------------------------------------------------------------------- */

/* Multiplicative mass corrector for REL bounds (requires single-sign data,
   which REL already requires). alpha = L0 / sum(xhat). Returns alpha. */
static inline double conserve_scale_mass(float* x, long N, double L0)
{
    double s = 0.0;
    for (long i = 0; i < N; i++) s += x[i];
    const double a = (s != 0.0) ? L0 / s : 1.0;
    for (long i = 0; i < N; i++) x[i] = (float)(a * (double)x[i]);
    return a;
}

/* Encoder-side helper (ABS): the largest pointwise displacement the mass+energy
   correction will add, computed on the DECOMPRESSED field xhat with the stored
   targets. Lets the encoder verify b' + maxdisp <= eps and tighten b' if not,
   without needing the original. Delta_i = (a-1)*(xhat_i - mean) + (mean0 - a*mean). */
static inline double conserve_mass_energy_maxdisp(const float* x, long N,
                                                  double M0, double S0)
{
    double s = 0.0, s2 = 0.0;
    for (long i = 0; i < N; i++) { double v = x[i]; s += v; s2 += v * v; }
    const double mean  = s  / (double)N;
    const double var   = s2 / (double)N - mean * mean;
    const double mean0 = M0 / (double)N;
    const double var0  = S0 / (double)N - mean0 * mean0;
    const double a = (var > 0.0) ? sqrt(var0 / var) : 1.0;
    const double off = mean0 - a * mean;
    double md = 0.0;
    for (long i = 0; i < N; i++) {
        double d = fabs((a - 1.0) * ((double)x[i] - mean) + off);
        if (d > md) md = d;
    }
    return md;
}

enum { CONS_BOUND_ABS = 0, CONS_BOUND_REL = 1 };

/* REL-safe dispatch. Applies the multiplicative corrector(s) that preserve a
   relative bound. Returns 1 on success; returns 0 (and does nothing) if asked
   for mass AND energy together, which is not a single global affine under REL. */
static inline int conserve_apply_rel(float* x, long N, int which,
                                     double M0, double S0)
{
    const int want_m = (which & CONS_MASS)   != 0;
    const int want_e = (which & CONS_ENERGY) != 0;
    if (want_m && want_e) return 0;                 /* not single-affine under REL */
    if (want_m) { conserve_scale_mass(x, N, M0); return 1; }
    if (want_e) {
        double s2 = 0.0; for (long i = 0; i < N; i++) { double v = x[i]; s2 += v*v; }
        const double a = (s2 > 0.0) ? sqrt(S0 / s2) : 1.0;
        for (long i = 0; i < N; i++) x[i] = (float)(a * (double)x[i]);
        return 1;
    }
    return 1;
}

/* ===========================================================================
   PRECOMPUTED PARAMETERS -- store (a,b) instead of the target invariants.
   If the codec already decompresses at encode time to verify its error bound
   (e.g. PFPL), it has xhat in hand and can compute the correction's scalars
   there, storing them as side-info in place of M0/S0/L0. The whole correction
   is a single affine map x <- a*xhat + b, so TWO scalars (a,b) cover every case
   (mass, momentum, energy, mass+energy). The decoder then does ONE elementwise
   pass and NO reduction -- on a GPU, a single FMA kernel with no device-wide
   reduction and no host round-trip.

   Encoder (has xhat plus M0/S0/L0 from the original): call the matching
   conserve_*_params to get (a,b).  Decoder: conserve_apply_affine(x,N,a,b).
   This composes with the ABS verify/tighten step for free: the encoder already
   has xhat and the original, so it can apply (a,b), check the corrected field
   against eps, and retighten before committing. Store (a,b) in double.
--------------------------------------------------------------------------- */

/* Decoder: single pass, no reduction. x[i] <- a*x[i] + b. */
static inline void conserve_apply_affine(float* x, long N, double a, double b)
{
    for (long i = 0; i < N; i++) x[i] = (float)(a * (double)x[i] + b);
}

/* Encoder, mass+energy -> (a,b): a = sqrt(var0/var), b = mean0 - a*mean.
   C_i = mean0 + a(xhat_i - mean) = a*xhat_i + b, identical to conserve_mass_energy. */
static inline void conserve_mass_energy_params(const float* x, long N,
                                               double M0, double S0,
                                               double* a_out, double* b_out)
{
    double s = 0.0, s2 = 0.0;
    for (long i = 0; i < N; i++) { double v = x[i]; s += v; s2 += v * v; }
    const double mean  = s  / (double)N;
    const double var   = s2 / (double)N - mean * mean;
    const double mean0 = M0 / (double)N;
    const double var0  = S0 / (double)N - mean0 * mean0;
    const double a = (var > 0.0) ? sqrt(var0 / var) : 1.0;
    *a_out = a; *b_out = mean0 - a * mean;
}

/* Encoder, mass or weighted momentum -> (1,b). w==NULL => unweighted mass. */
static inline void conserve_linear_params(const float* x, long N, double L0,
                                          const float* w,
                                          double* a_out, double* b_out)
{
    double swx = 0.0, sw = 0.0;
    if (w) { for (long i = 0; i < N; i++) { swx += (double)w[i]*x[i]; sw += (double)w[i]; } }
    else   { for (long i = 0; i < N; i++)   swx += x[i];             sw = (double)N;      }
    *a_out = 1.0; *b_out = (sw != 0.0) ? (L0 - swx) / sw : 0.0;
}

/* Encoder, energy only -> (a,0): a = sqrt(S0 / sum xhat^2). */
static inline void conserve_energy_params(const float* x, long N, double S0,
                                          double* a_out, double* b_out)
{
    double s2 = 0.0; for (long i = 0; i < N; i++) { double v = x[i]; s2 += v*v; }
    *a_out = (s2 > 0.0) ? sqrt(S0 / s2) : 1.0; *b_out = 0.0;
}

/* ---------------------------------------------------------------------------
   Multiple invariants.
   - Invariants on SEPARATE fields (e.g. mass on rho, momentum on each of
     rho*v_x, v_y, v_z) are independent: apply the matching routine to each
     field. No coupling.
   - Two invariants on the SAME field that do not decouple (e.g. energy =
     sum(x^2) AND momentum = sum(x) enforced together with a single affine
     a*x+b) are a 2x2 solve for (a,b):
         sum(a*x+b)     = M0        ->  a*Sx + b*N          = M0
         sum((a*x+b)^2) = S0        ->  a^2*Sxx + 2ab*Sx + b^2*N = S0
     Substitute b=(M0 - a*Sx)/N into the second -> quadratic in a; take the
     root nearest 1. (The decoupled mass+energy routine above is the special
     case that is always real because it scales about the mean; prefer it
     unless a strict single-affine form is required.)
   The general rule: linear invariant -> offset; quadratic invariant ->
   scale-about-mean; k invariants on one field -> a k-parameter solve.
--------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------------
   CUDA sketch for conserve_mass_energy (per field/block):
     // 1) two reductions (e.g. cub::DeviceReduce or a fused custom kernel):
     //      s  = sum(x[i]);  s2 = sum(x[i]*x[i]);
     // 2) on host: mean, var, mean0, var0, a  (as above)
     // 3) one elementwise kernel:
     //      x[i] = mean0 + a*(x[i] - mean);
   Cost: O(N) memory traffic, two passes (or one fused reduce + one map);
   negligible next to the codec. Do reductions in double or Kahan-compensated.
--------------------------------------------------------------------------- */
#endif /* CONSERVE_H */
