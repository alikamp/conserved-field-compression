/*
  conserve_cuda.h  —  Reference conservation corrector for lossy scientific codecs.
  A. M. Parks + M. Burtscher, 2026.  Public reference implementation for integration into
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

#ifndef CONSERVE_CUDA_H
#define CONSERVE_CUDA_H
#include <math.h>

static constexpr const int TPB = 512;  // threads per block

static __global__ void c_sum(const float* const __restrict__ x, const long long N, double* const __restrict__ res)
{
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int incr = gridDim.x * blockDim.x;

  double val = 0.0;
  for (long long i = start; i < N; i += incr) {
    const double v = x[i];
    val += v;
  }

  val += __shfl_xor_sync(~0, val, 1);
  val += __shfl_xor_sync(~0, val, 2);
  val += __shfl_xor_sync(~0, val, 4);
  val += __shfl_xor_sync(~0, val, 8);
  val += __shfl_xor_sync(~0, val, 16);

  if ((threadIdx.x % 32) == 0) {
    atomicAdd(&res[0], val);
  }
}

static __global__ void c_sum2(const float* const __restrict__ x, const long long N, double* const __restrict__ res)
{
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int incr = gridDim.x * blockDim.x;

  double val2 = 0.0;
  for (long long i = start; i < N; i += incr) {
    const double v = x[i];
    val2 += v * v;
  }

  val2 += __shfl_xor_sync(~0, val2, 1);
  val2 += __shfl_xor_sync(~0, val2, 2);
  val2 += __shfl_xor_sync(~0, val2, 4);
  val2 += __shfl_xor_sync(~0, val2, 8);
  val2 += __shfl_xor_sync(~0, val2, 16);

  if ((threadIdx.x % 32) == 0) {
    atomicAdd(&res[0], val2);
  }
}

static __global__ void c_sum_sum2(const float* const __restrict__ x, const long long N, double* const __restrict__ res)
{
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int incr = gridDim.x * blockDim.x;

  double val = 0.0;
  double val2 = 0.0;
  for (long long i = start; i < N; i += incr) {
    const double v = x[i];
    val += v;
    val2 += v * v;
  }

  val += __shfl_xor_sync(~0, val, 1);
  val2 += __shfl_xor_sync(~0, val2, 1);
  val += __shfl_xor_sync(~0, val, 2);
  val2 += __shfl_xor_sync(~0, val2, 2);
  val += __shfl_xor_sync(~0, val, 4);
  val2 += __shfl_xor_sync(~0, val2, 4);
  val += __shfl_xor_sync(~0, val, 8);
  val2 += __shfl_xor_sync(~0, val2, 8);
  val += __shfl_xor_sync(~0, val, 16);
  val2 += __shfl_xor_sync(~0, val2, 16);

  if ((threadIdx.x % 32) == 0) {
    atomicAdd(&res[0], val);
    atomicAdd(&res[1], val2);
  }
}

static __global__ void c_wsum_sum(const float* const __restrict__ x, const long long N, const float* const __restrict__ w, double* const __restrict__ res)
{
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int incr = gridDim.x * blockDim.x;

  double val = 0.0;
  double val2 = 0.0;
  for (long long i = start; i < N; i += incr) {
    const double v = x[i];
    const double u = w[i];
    val += v * u;
    val2 += u;
  }

  val += __shfl_xor_sync(~0, val, 1);
  val2 += __shfl_xor_sync(~0, val2, 1);
  val += __shfl_xor_sync(~0, val, 2);
  val2 += __shfl_xor_sync(~0, val2, 2);
  val += __shfl_xor_sync(~0, val, 4);
  val2 += __shfl_xor_sync(~0, val2, 4);
  val += __shfl_xor_sync(~0, val, 8);
  val2 += __shfl_xor_sync(~0, val2, 8);
  val += __shfl_xor_sync(~0, val, 16);
  val2 += __shfl_xor_sync(~0, val2, 16);

  if ((threadIdx.x % 32) == 0) {
    atomicAdd(&res[0], val);
    atomicAdd(&res[1], val2);
  }
}

static __global__ void c_mul_map(float* const __restrict__ x, const long long N, const double* const __restrict__ res)
{
  const double r1 = res[1];
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int incr = gridDim.x * blockDim.x;
  for (long long i = start; i < N; i += incr) {
    x[i] = (float)(r1 * (double)x[i]);
  }
}

static inline void setBlocks(int& blocks)
{
  if (blocks == 0) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    //if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); exit(-1);}
    const int SMs = deviceProp.multiProcessorCount;
    const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
    blocks = SMs * (mTpSM / TPB);
  }
}

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

static __global__ void cme_mean_var(const long long N, const double M0, const double S0, double* const __restrict__ res)
{
  const double mean = res[0] / N;
  const double mean0 = M0 / N;
  const double var = res[1] / N - mean * mean;
  const double var0 = S0 / N - mean0 * mean0;
  const double a = (var > 0.0) ? sqrt(var0 / var) : 1.0;
  res[2] = mean;
  res[3] = mean0;
  res[4] = a;
}

static __global__ void cme_map(float* const __restrict__ x, const long long N, const double* const __restrict__ res)
{
  const double r2 = res[2];
  const double r3 = res[3];
  const double r4 = res[4];
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int incr = gridDim.x * blockDim.x;
  for (long long i = start; i < N; i += incr) {
    x[i] = (float)(r3 + r4 * ((double)x[i] - r2));
  }
}

static inline void conserve_mass_energy(float* const __restrict__ d_x, const long long N, const double M0, const double S0, int blocks = 0)
{
  setBlocks(blocks);

  double* d_res;
  cudaMalloc(&d_res, 5 * sizeof(double));
  cudaMemsetAsync(d_res, 0, 2 * sizeof(double));

  c_sum_sum2<<<blocks, TPB>>>(d_x, N, d_res);
  cme_mean_var<<<1, 1>>>(N, M0, S0, d_res);
  cme_map<<<blocks, TPB>>>(d_x, N, d_res);

  cudaFree(d_res);
}

/* ---------------------------------------------------------------------------
   Any single LINEAR invariant  L0 = sum(w*x)  (mass: w=1; momentum on a
   component: w = density, or w=1 for a velocity-sum convention).  Offset only.
   For w=1 this is mass; pass a weight array for a weighted linear invariant.
   Side-info: L0 (one scalar).  Offset does not perturb any energy term that
   depends on differences of x, so it composes with the energy scale above.
--------------------------------------------------------------------------- */

static __global__ void cl_b(const long long N, const double L0, double* const __restrict__ res)
{
  res[1] = (L0 - res[0]) / N;
}

static __global__ void cl_map(float* const __restrict__ x, const long long N, const double* const __restrict__ res)
{
  const double r1 = res[1];
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int incr = gridDim.x * blockDim.x;
  for (long long i = start; i < N; i += incr) {
    x[i] = (float)((double)x[i] + r1);
  }
}

static inline void conserve_linear(float* const __restrict__ d_x, const long long N, const double L0, int blocks = 0)
{
  setBlocks(blocks);

  double* d_res;
  cudaMalloc(&d_res, 2 * sizeof(double));
  cudaMemsetAsync(d_res, 0, 1 * sizeof(double));

  c_sum<<<blocks, TPB>>>(d_x, N, d_res);
  cl_b<<<1, 1>>>(N, L0, d_res);
  cl_map<<<blocks, TPB>>>(d_x, N, d_res);

  cudaFree(d_res);
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

static __global__ void clw_b(const double L0, double* const __restrict__ res)
{
  const double sw = res[1];
  res[2] = (sw != 0.0) ? ((L0 - res[0]) / sw) : 0.0;
}

static __global__ void clw_map(float* const __restrict__ x, const long long N, const double* const __restrict__ res)
{
  const double r2 = res[2];
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int incr = gridDim.x * blockDim.x;
  for (long long i = start; i < N; i += incr) {
    x[i] = (float)((double)x[i] + r2);
  }
}

static inline void conserve_linear_w(float* const __restrict__ d_x, const long long N, const double L0, const float* const __restrict__ d_w, int blocks = 0)
{
  if (!d_w) { conserve_linear(d_x, N, L0); return; }

  setBlocks(blocks);

  double* d_res;
  cudaMalloc(&d_res, 3 * sizeof(double));
  cudaMemsetAsync(d_res, 0, 2 * sizeof(double));

  c_wsum_sum<<<blocks, TPB>>>(d_x, N, d_w, d_res);
  clw_b<<<1, 1>>>(L0, d_res);
  clw_map<<<blocks, TPB>>>(d_x, N, d_res);

  cudaFree(d_res);
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

static __global__ void ce_a(const long long N, const double S0, double* const __restrict__ res)
{
  const double s2 = res[0];
  res[1] = (s2 > 0.0) ? sqrt(S0 / s2) : 1.0;
}

static inline void conserve_energy(float* const __restrict__ d_x, const long long N, const double S0, int blocks = 0)
{
  setBlocks(blocks);

  double* d_res;
  cudaMalloc(&d_res, 2 * sizeof(double));
  cudaMemsetAsync(d_res, 0, 1 * sizeof(double));

  c_sum2<<<blocks, TPB>>>(d_x, N, d_res);
  ce_a<<<1, 1>>>(N, S0, d_res);
  c_mul_map<<<blocks, TPB>>>(d_x, N, d_res);

  cudaFree(d_res);
}

/* which: bitmask of invariants to preserve (user option, set per job).
   M0,S0: the stored targets named by `which`.
   w:     optional weight field for the linear invariant (momentum: w=density;
          NULL = unweighted, i.e. mass). Ignored for the energy-only path. */

static inline void conserve_apply(float* const __restrict__ d_x, const long long N, const int which, const double M0, const double S0, const float* const d_w, int blocks = 0)
{
  const int want_m = ((which & CONS_MASS) != 0);
  const int want_e = ((which & CONS_ENERGY) != 0);
  if (want_m && want_e) { conserve_mass_energy(d_x, N, M0, S0, blocks); return; }  /* offset + scale */
  if (want_m) { conserve_linear_w(d_x, N, M0, d_w, blocks); return; }  /* mass/momentum: (weighted) offset */
  if (want_e) { conserve_energy(d_x, N, S0, blocks); }  /* energy only: global scale */
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

static __global__ void csm_a(const long long N, const double L0, double* const __restrict__ res)
{
  const double s = res[0];
  res[1] = (s != 0.0) ? (L0 / s) : 1.0;
}

static inline void conserve_scale_mass(float* const __restrict__ d_x, const long long N, const double L0, int blocks = 0)
{
  setBlocks(blocks);

  double* d_res;
  cudaMalloc(&d_res, 2 * sizeof(double));
  cudaMemsetAsync(d_res, 0, 1 * sizeof(double));

  c_sum<<<blocks, TPB>>>(d_x, N, d_res);
  csm_a<<<1, 1>>>(N, L0, d_res);
  c_mul_map<<<blocks, TPB>>>(d_x, N, d_res);

  cudaFree(d_res);
}

/* Encoder-side helper (ABS): the largest pointwise displacement the mass+energy
   correction will add, computed on the DECOMPRESSED field xhat with the stored
   targets. Lets the encoder verify b' + maxdisp <= eps and tighten b' if not,
   without needing the original. Delta_i = (a-1)*(xhat_i - mean) + (mean0 - a*mean). */

static __global__ void cmem_mean_var(const long long N, const double M0, const double S0, double* const  __restrict__ res)
{
  const double mean = res[0] / N;
  const double mean0 = M0 / N;
  const double var = res[1] / N - mean * mean;
  const double var0 = S0 / N - mean0 * mean0;
  const double a = (var > 0.0) ? sqrt(var0 / var) : 1.0;
  res[3] = mean;
  res[4] = a - 1.0;
  res[5] = mean0 - a * mean;
}

static __global__ void cmem_max(const float* const __restrict__ x, const long long N, double* const __restrict__ res)
{
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int incr = gridDim.x * blockDim.x;

  const double r3 = res[3];
  const double r4 = res[4];
  const double r5 = res[5];
  double val = 0.0;
  for (long long i = start; i < N; i += incr) {
    const double d = fabs(r4 * ((double)x[i] - r3) + r5);
    val = max(val, d);
  }

  val = max(val, __shfl_xor_sync(~0, val, 1));
  val = max(val, __shfl_xor_sync(~0, val, 2));
  val = max(val, __shfl_xor_sync(~0, val, 4));
  val = max(val, __shfl_xor_sync(~0, val, 8));
  val = max(val, __shfl_xor_sync(~0, val, 16));

  if ((threadIdx.x % 32) == 0) {
    atomicMax((long long*)&res[2], *((long long*)&val));  // hack, but works for positive doubles
  }
}

static inline double conserve_mass_energy_maxdisp(const float* const __restrict__ d_x, const long long N, const double M0, const double S0, int blocks = 0)
{
  setBlocks(blocks);

  double* d_res;
  cudaMalloc(&d_res, 6 * sizeof(double));
  cudaMemsetAsync(d_res, 0, 3 * sizeof(double));

  c_sum_sum2<<<blocks, TPB>>>(d_x, N, d_res);
  cmem_mean_var<<<1, 1>>>(N, M0, S0, d_res);
  cmem_max<<<blocks, TPB>>>(d_x, N, d_res);

  double md;
  cudaMemcpy(&md, &d_res[2], sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_res);
  return md;
}

enum { CONS_BOUND_ABS = 0, CONS_BOUND_REL = 1 };

/* REL-safe dispatch. Applies the multiplicative corrector(s) that preserve a
   relative bound. Returns 1 on success; returns 0 (and does nothing) if asked
   for mass AND energy together, which is not a single global affine under REL. */

static __global__ void cse_a(const long long N, const double S0, double* const  __restrict__ res)
{
  const double s2 = res[0];
  res[1] = (s2 > 0.0) ? sqrt(S0 / s2) : 1.0;
}

static inline void conserve_scale_energy(float* const __restrict__ d_x, const long long N, const double S0, int blocks = 0)
{
  setBlocks(blocks);

  double* d_res;
  cudaMalloc(&d_res, 2 * sizeof(double));
  cudaMemsetAsync(d_res, 0, 1 * sizeof(double));

  c_sum2<<<blocks, TPB>>>(d_x, N, d_res);
  cse_a<<<1, 1>>>(N, S0, d_res);
  c_mul_map<<<blocks, TPB>>>(d_x, N, d_res);

  cudaFree(d_res);
}

static inline int conserve_apply_rel(float* const __restrict__ d_x, const long long N, const int which, const double M0, const double S0)
{
  const int want_m = ((which & CONS_MASS) != 0);
  const int want_e = ((which & CONS_ENERGY) != 0);
  if (want_m && want_e) return 0;  /* not single-affine under REL */
  if (want_m) { conserve_scale_mass(d_x, N, M0); return 1; }
  if (want_e) { conserve_scale_energy(d_x, N, S0); return 1; }
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

static __global__ void caa_map(float* const __restrict__ x, const long long N, const double a, const double b)
{
  const long long tid = blockIdx.x * (long long)TPB + threadIdx.x;
  if (tid < N) {
    x[tid] = (float)(a * (double)x[tid] + b);
  }
}

static inline void conserve_apply_affine(float* const d_x, const long long N, const double a, const double b)
{
  caa_map<<<(N + TPB - 1) / TPB, TPB>>>(d_x, N, a, b);
  cudaDeviceSynchronize();
}

/* Encoder, mass+energy -> (a,b): a = sqrt(var0/var), b = mean0 - a*mean.
   C_i = mean0 + a(xhat_i - mean) = a*xhat_i + b, identical to conserve_mass_energy. */

static inline void conserve_mass_energy_params(const float* const __restrict__ d_x, const long long N, const double M0, const double S0, double& a_out, double& b_out, int blocks = 0)
{
  setBlocks(blocks);

  double* d_res;
  cudaMalloc(&d_res, 5 * sizeof(double));
  cudaMemsetAsync(d_res, 0, 2 * sizeof(double));

  c_sum_sum2<<<blocks, TPB>>>(d_x, N, d_res);
  cme_mean_var<<<1, 1>>>(N, M0, S0, d_res);

  double res [5];
  cudaMemcpy(&res[2], &d_res[2], 3 * sizeof(double), cudaMemcpyDeviceToHost);
  a_out = res[4];
  b_out = res[3] - a_out * res[2];

  cudaFree(d_res);
}

/* Encoder, mass or weighted momentum -> (1,b). w==NULL => unweighted mass. */

static inline void conserve_linear_params(const float* const __restrict__ d_x, const long long N, const double L0, const float* const __restrict__ d_w, double& a_out, double& b_out, int blocks = 0)
{
  setBlocks(blocks);

  double* d_res;
  if (d_w) {
    cudaMalloc(&d_res, 3 * sizeof(double));
    cudaMemsetAsync(d_res, 0, 2 * sizeof(double));

    c_wsum_sum<<<blocks, TPB>>>(d_x, N, d_w, d_res);
    clw_b<<<1, 1>>>(L0, d_res);

    cudaMemcpy(&b_out, &d_res[2], sizeof(double), cudaMemcpyDeviceToHost);
  } else {
    cudaMalloc(&d_res, 2 * sizeof(double));
    cudaMemsetAsync(d_res, 0, 1 * sizeof(double));

    c_sum<<<blocks, TPB>>>(d_x, N, d_res);
    cl_b<<<1, 1>>>(N, L0, d_res);

    cudaMemcpy(&b_out, &d_res[1], sizeof(double), cudaMemcpyDeviceToHost);
  }
  a_out = 1.0;

  cudaFree(d_res);
}

/* Encoder, energy only -> (a,0): a = sqrt(S0 / sum xhat^2). */

static inline void conserve_energy_params(const float* const __restrict__ d_x, const long long N, const double S0, double& a_out, double& b_out, int blocks = 0)
{
  setBlocks(blocks);

  double* d_res;
  cudaMalloc(&d_res, 2 * sizeof(double));
  cudaMemsetAsync(d_res, 0, 1 * sizeof(double));

  c_sum2<<<blocks, TPB>>>(d_x, N, d_res);
  ce_a<<<1, 1>>>(N, S0, d_res);

  cudaMemcpy(&a_out, &d_res[1], sizeof(double), cudaMemcpyDeviceToHost);
  b_out = 0.0;

  cudaFree(d_res);
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

#endif /* CONSERVE_CUDA_H */
