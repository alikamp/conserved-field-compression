/* fp32_test.cpp -- can the corrector run without FP64 on the GPU?
   CPU emulation of a persistent-thread GPU layout (T threads, grid-stride loop,
   per-thread partials combined at the end). Compares the restored mass/energy drift of
     reductions:  FP64 (current) | FP32 plain | FP32 Kahan per thread, FP64 only to combine partials
     apply pass:  FP64 a*x+b     | FP32 x + fmaf(da, x, g), da = a-1 and g = b computed once in FP64
                  | FP32 fmaf((float)a, x, g)  (shows why a-1 must be passed, not a)
   Products are FP32 in every FP32 variant.
   Build: g++ -O2 -ffp-contract=off fp32_test.cpp -o fp32_test && ./fp32_test        */
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

struct Sums { double M, S; };
static const long T = 32768;                     /* persistent threads */

static Sums exact(const std::vector<float>& x) { /* long-double reference */
  long double m = 0, s = 0; for (float v : x) { m += v; s += (long double)v * v; } return {(double)m, (double)s}; }

static Sums red_fp64(const std::vector<float>& x) {
  long N = x.size(); double M = 0, S = 0;
  for (long t = 0; t < T; t++) { double m = 0, s = 0;
    for (long i = t; i < N; i += T) { double v = x[i]; m += v; s += v * v; } M += m; S += s; }
  return {M, S}; }

static Sums red_fp32(const std::vector<float>& x) {
  long N = x.size(); double M = 0, S = 0;
  for (long t = 0; t < T; t++) { float m = 0, s = 0;
    for (long i = t; i < N; i += T) { float v = x[i]; m += v; s += v * v; } M += m; S += s; }
  return {M, S}; }

static Sums red_fp32_kahan(const std::vector<float>& x) {
  long N = x.size(); double M = 0, S = 0;
  for (long t = 0; t < T; t++) { float m = 0, cm = 0, s = 0, cs = 0;
    for (long i = t; i < N; i += T) { float v = x[i];
      float y  = v - cm;     float t1 = m + y;  cm = (t1 - m) - y;  m = t1;   /* Kahan, FP32 */
      float p  = v * v;      float y2 = p - cs; float t2 = s + y2; cs = (t2 - s) - y2; s = t2; }
    M += (double)m - (double)cm; S += (double)s - (double)cs; }               /* FP64: once per thread */
  return {M, S}; }

static void params(Sums h, double M0, double S0, long N, double& a, double& b) {
  double mean = h.M / N, var = h.S / N - mean * mean, mean0 = M0 / N, var0 = S0 / N - mean0 * mean0;
  a = sqrt(var0 / var); b = mean0 - a * mean; }

static void run(const char* name, const std::vector<float>& x, std::mt19937& g) {
  long N = x.size(); Sums o = exact(x);
  float mn = x[0], mx = x[0]; for (float v : x) { mn = fminf(mn, v); mx = fmaxf(mx, v); }
  double bnd = 1e-3 * (mx - mn);                 /* stand-in codec: uniform errors, NOA 1e-3 */
  std::uniform_real_distribution<double> U(-bnd, bnd); std::vector<float> xh(N);
  for (long i = 0; i < N; i++) xh[i] = (float)(x[i] + U(g));
  Sums h = exact(xh);
  auto dr = [&](Sums r, char* buf) { sprintf(buf, "M %.1e  E %.1e", fabs(r.M - o.M) / fabs(o.M), fabs(r.S - o.S) / o.S); };
  char b0[64]; dr(h, b0); printf("\n%s (N=%ld)   raw drift: %s\n", name, N, b0);
  printf("  %-24s %-22s %-22s %-22s\n", "reduction", "apply FP64", "apply FP32 (a-1)", "apply FP32 (float a)");
  const char* nm[] = {"FP64 (current)", "FP32 plain", "FP32 Kahan, FP64 combine"};
  for (int k = 0; k < 3; k++) {
    auto red = [&](const std::vector<float>& y) { return k == 0 ? red_fp64(y) : k == 1 ? red_fp32(y) : red_fp32_kahan(y); };
    Sums e = red(x), d = red(xh);                /* encoder: sums of x;  decoder: sums of xhat */
    double a, b; params(d, e.M, e.S, N, a, b);
    float da = (float)(a - 1.0), gb = (float)b, af = (float)a;
    std::vector<float> c1(N), c2(N), c3(N);
    for (long i = 0; i < N; i++) { c1[i] = (float)(a * xh[i] + b);
                                   c2[i] = xh[i] + fmaf(da, xh[i], gb);
                                   c3[i] = fmaf(af, xh[i], gb); }
    char s1[64], s2[64], s3[64]; dr(exact(c1), s1); dr(exact(c2), s2); dr(exact(c3), s3);
    printf("  %-24s %-22s %-22s %-22s\n", nm[k], s1, s2, s3);
  }
}

int main() {
  std::mt19937 g(3); std::normal_distribution<double> Z(0, 1); long N = 1L << 25; std::vector<float> x(N);
  for (long i = 0; i < N; i++) { double t = (double)i / N; x[i] = (float)(sin(40 * t) + 0.3 * Z(g)); }
  run("signed, near-zero mean", x, g);
  for (long i = 0; i < N; i++) { double t = (double)i / N; x[i] = (float)(280 + 20 * sin(40 * t) + 3 * Z(g)); }
  run("large offset (temperature-like)", x, g);
  for (long i = 0; i < N; i++) x[i] = (float)exp(Z(g));
  run("lognormal", x, g);
  return 0;
}
