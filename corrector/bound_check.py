"""
Does the conservation layer preserve the pointwise bound? Test ABS and REL
separately, isolating the CORRECTOR (inject codec error that already respects
the bound, then correct, then re-measure the bound). Answers Martin's Oct 5 Q.

ABS:  |x - x_hat| <= b           (additive)
REL:  |1 - x_hat/x| <= r, same sign   (multiplicative, enforced in log space)
"""
import numpy as np
rng = np.random.default_rng(0)

# A field with (a) a bulk, (b) large-deviation outliers, (c) near-zero values.
N = 200000
bulk = rng.normal(10.0, 1.0, N)                      # mean ~10, sigma ~1
bulk[:2000] = rng.normal(10.0, 1.0, 2000) + rng.choice([-1,1],2000)*rng.uniform(6,9,2000)  # outliers far from mean
x_abs = bulk.copy()                                  # zero-free, has far-from-mean points
# single-sign field with values spanning down to ~0 for the REL test
x_rel = np.abs(rng.normal(0.0, 1.0, N))**2 + 1e-6    # many values near zero
x_rel = x_rel.astype(np.float64)

def invs(f): d=f.astype(np.float64); return d.sum(), (d*d).sum()

# ---- correctors (match conserve.h) ----
def c_linear(xh, L0):                       # additive offset -> mass
    return xh + (L0 - xh.sum())/xh.size
def c_scale(xh, S0):                         # global multiplicative scale -> energy
    s2 = (xh*xh).sum(); a = np.sqrt(S0/s2) if s2>0 else 1.0
    return a*xh, a
def c_scale_mass(xh, L0):                    # global multiplicative scale -> mass (single sign)
    s = xh.sum(); a = L0/s if s!=0 else 1.0
    return a*xh, a
def c_mass_energy(xh, M0, S0):               # offset + scale-about-mean
    Nn=xh.size; mean=xh.mean(); var=xh.var()
    mean0=M0/Nn; var0=max(S0/Nn-mean0*mean0,0.0)
    a=np.sqrt(var0/var) if var>0 else 1.0
    return (mean0 + a*(xh-mean)), a

print("="*74)
print("ABS test (requested bound B; compress at b'=(1-eps)B):  x zero-free")
print("="*74)
x = x_abs; M0,S0 = invs(x)
B = 0.01*(x.max()-x.min())
for eps in (0.10,):
    bp = (1-eps)*B
    e = rng.uniform(-bp, bp, N); xh = x + e            # codec error, respects b'
    # mass-only (linear / additive)
    cl = c_linear(xh, M0)
    beta = (M0 - xh.sum())/N
    max_lin = np.abs(cl - x).max()
    # mass+energy (affine)
    cme, a = c_mass_energy(xh, M0, S0)
    max_me = np.abs(cme - x).max()
    # where the ME violation lives: correlate excess with distance from mean
    excess = np.abs(cme - x)
    far = np.argsort(np.abs(x - x.mean()))[-5:]
    print(f"eps={eps}  B={B:.4g}  b'={bp:.4g}")
    print(f"  |beta| (offset)          = {abs(beta):.4g}   (<= b' = {bp:.4g})")
    print(f"  a-1  (energy multiplier) = {a-1:.3e}")
    print(f"  conserve_linear      max|C-x| = {max_lin:.4g}   -> /B = {max_lin/B:.3f}  {'OK' if max_lin<=B else 'VIOLATES B'}")
    print(f"  conserve_mass_energy max|C-x| = {max_me:.4g}   -> /B = {max_me/B:.3f}  {'OK' if max_me<=B else 'VIOLATES B'}")
    print(f"  ME worst points are far-from-mean: |x-mean| at 5 worst-excess pts =",
          np.round(np.abs(x-x.mean())[np.argsort(excess)[-5:]],2))
    print(f"  predicted extra at max-dev pt |a-1|*max|x-mean| = {abs(a-1)*np.abs(x-x.mean()).max():.4g}")

print()
print("="*74)
print("REL test (requested bound r):  x single-sign with near-zero values")
print("="*74)
x = x_rel; M0,S0 = invs(x)
r = 1e-3
for eps in (0.10,):
    rp = (1-eps)*r
    rho = rng.uniform(-rp, rp, N); xh = x*(1+rho)       # codec error, respects REL r'
    relraw = np.abs(1 - xh/x).max()
    # additive mass offset
    cl = c_linear(xh, M0)
    rel_lin = np.abs(1 - cl/x).max()
    i_lin = np.argmax(np.abs(1 - cl/x))
    # multiplicative energy scale
    ce, ae = c_scale(xh, S0)
    rel_e = np.abs(1 - ce/x).max()
    # multiplicative mass scale
    cm, am = c_scale_mass(xh, M0)
    rel_m = np.abs(1 - cm/x).max()
    print(f"r={r}  r'={rp:.4g}   raw max rel after codec = {relraw:.4g}")
    print(f"  conserve_linear (additive mass) max rel = {rel_lin:.4g}  -> /r = {rel_lin/r:.3g}  {'OK' if rel_lin<=r else 'VIOLATES r'}")
    print(f"      worst point x = {x[i_lin]:.3e} (near zero); offset beta = {(M0-xh.sum())/N:.3e}")
    print(f"  global SCALE for energy (mult.)  max rel = {rel_e:.4g}  -> /r = {rel_e/r:.3g}  {'OK' if rel_e<=r else 'VIOLATES r'}   a-1={ae-1:.2e}")
    print(f"  global SCALE for mass   (mult.)  max rel = {rel_m:.4g}  -> /r = {rel_m/r:.3g}  {'OK' if rel_m<=r else 'VIOLATES r'}   a-1={am-1:.2e}")
