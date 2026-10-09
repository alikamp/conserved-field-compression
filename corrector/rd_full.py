"""
rd_full.py -- Alika Parks
Systematic storage-case rate-distortion harness for the conservation layer.
Backbone table for the IWBDR-6 paper (E1/E2): real coded ratios per codec, with
invariant drift (mass, energy) layer-on/off, pointwise-bound check, and PSNR.

Codecs:
  - ZFP           via zfpy                       (works anywhere)
  - SZ3, MGARD,   via libpressio                 (Colab: condacolab + conda-forge)
    ZFP ROUND_FIRST
The conservation layer matches conserve.h (decoupled affine, always real).
Bound handling (per Burtscher): compress at a uniformly tightened bound
b' = (1-eps)*b so the corrector's rescale stays within the user's bound b.

Usage:
  python rd_full.py                      # self-test: ZFP on a synthetic 3D field + its 2D slice + 1D line-out
  python rd_full.py field.npy [name]     # run the sweep on a real field (any dim)
"""
import sys, numpy as np

# ---------- conservation layer (== conserve.h) ------------------------------
def invariants(f):
    d = f.astype(np.float64); return float(d.sum()), float((d*d).sum())

def conserve(fh, M0, S0):
    hd = fh.astype(np.float64); N = hd.size
    mean, var = hd.mean(), hd.var()
    mean0 = M0/N; var0 = max(S0/N - mean0*mean0, 0.0)
    a = np.sqrt(var0/var) if var > 0 else 1.0
    return (mean0 + a*(hd - mean)).astype(np.float32)

def conserve_linear(fh, L0):                 # E4: a linear invariant (momentum comp.)
    hd = fh.astype(np.float64)
    return (hd + (L0 - hd.sum())/hd.size).astype(np.float32)

# ---------- codec adapters --------------------------------------------------
class ZFP:
    name = "zfp"
    def compress(self, field, abs_tol):
        import zfpy
        c = zfpy.compress_numpy(np.ascontiguousarray(field), tolerance=abs_tol)
        return len(c), zfpy.decompress_numpy(c)

class Libpressio:
    """Colab path. compressor in {'sz3','mgard','zfp'}; round_first for ZFP."""
    def __init__(self, compressor="sz3", round_first=False):
        self.compressor = compressor; self.round_first = round_first
        self.name = compressor + ("+roundfirst" if round_first else "")
    def compress(self, field, abs_tol):
        from libpressio import PressioCompressor
        opts = {"pressio:abs": float(abs_tol)}
        early = {}
        if self.compressor == "zfp" and self.round_first:
            early = {"zfp:exec_name": "serial"}  # ROUND_FIRST set via zfp:round; see note
            opts["zfp:round"] = 1                 # ZFP_ROUND_FIRST
        comp = PressioCompressor.from_config({
            "compressor_id": self.compressor,
            "early_config": early,
            "compressor_config": opts,
        })
        field = np.ascontiguousarray(field)
        comp_data = comp.encode(field)
        rec = np.zeros_like(field); rec = comp.decode(comp_data, rec)
        nbytes = int(comp.get_metrics().get("size:compressed_size", comp_data.nbytes))
        return nbytes, rec

# ---------- driver ----------------------------------------------------------
def study(field, name, codec, fracs=(1e-2,3e-3,1e-3,3e-4,1e-4), eps=0.10):
    field = np.ascontiguousarray(field.astype(np.float32))
    N = field.size; rng = float(field.max()-field.min())
    M0, S0 = invariants(field); fd = field.astype(np.float64)
    rows = []
    for frac in fracs:
        b = frac*rng; bt = (1-eps)*b                      # user bound b, tightened b'
        nbytes, rec = codec.compress(field, bt)
        rd = rec.astype(np.float64)
        ratio = field.nbytes/nbytes; bpv = 8*nbytes/N
        maxerr_raw = float(np.abs(fd-rd).max())
        Mh, Sh = invariants(rec)
        fc = conserve(rec, M0, S0); cd = fc.astype(np.float64)
        maxerr_cor = float(np.abs(fd-cd).max())
        Mc, Sc = invariants(fc)
        mse = float(np.mean((fd-rd)**2)); psnr = 20*np.log10(rng)-10*np.log10(mse) if mse>0 else float("inf")
        rel = lambda x,x0: abs(x-x0)/abs(x0) if abs(x0)>1e-12 else float("nan")
        rows.append(dict(codec=codec.name, field=name, dim=field.ndim, frac=frac,
            ratio=ratio, bpv=bpv, psnr=psnr,
            maxerr_raw_over_b=maxerr_raw/b, maxerr_cor_over_b=maxerr_cor/b,
            E_raw=rel(Sh,S0), E_cor=rel(Sc,S0), M_raw=rel(Mh,M0), M_cor=rel(Mc,M0)))
    return rows

def print_rows(rows):
    h=f"{'codec':<14}{'field':<10}{'d':>2}{'frac':>7}{'ratio':>7}{'PSNR':>7}{'mxE/b':>7}{'mxEc/b':>7}{'E raw':>10}{'E cor':>10}{'M raw':>10}{'M cor':>10}"
    print(h); print('-'*len(h))
    for r in rows:
        print(f"{r['codec']:<14}{r['field']:<10}{r['dim']:>2}{r['frac']:>7.0e}{r['ratio']:>7.1f}"
              f"{r['psnr']:>7.1f}{r['maxerr_raw_over_b']:>7.2f}{r['maxerr_cor_over_b']:>7.2f}"
              f"{r['E_raw']:>10.2e}{r['E_cor']:>10.2e}{r['M_raw']:>10.2e}{r['M_cor']:>10.2e}")

def lineout(f3d):            # 1D scientific series: center line-out along axis 0
    a,b = f3d.shape[1]//2, f3d.shape[2]//2
    return np.ascontiguousarray(f3d[:, a, b])

def synth3d(n=64, seed=0):
    rng=np.random.default_rng(seed); fk=np.zeros((n,n,n),complex)
    fk[:6,:6,:6]=rng.standard_normal((6,6,6))+1j*rng.standard_normal((6,6,6))
    bg=np.fft.ifftn(fk).real; bg/=bg.std()
    xs=np.linspace(-1,1,n); X,Y,Z=np.meshgrid(xs,xs,xs,indexing='ij'); f=bg.copy()
    for _ in range(6):
        c=rng.uniform(-.6,.6,3); w=rng.uniform(.05,.15); amp=rng.uniform(1,4)
        f+=amp*np.exp(-((X-c[0])**2+(Y-c[1])**2+(Z-c[2])**2)/(2*w**2))
    return f.astype(np.float32)

if __name__ == "__main__":
    codec = ZFP()
    if len(sys.argv) > 1:
        f = np.load(sys.argv[1]).astype(np.float32)
        name = sys.argv[2] if len(sys.argv)>2 else "field"
        print_rows(study(f, name, codec))
    else:
        f3 = synth3d(64)
        rows  = study(f3,            "synth3d", codec)
        rows += study(f3[32],        "synth2d", codec)   # a 2D slice
        rows += study(lineout(f3),   "synth1d", codec)   # a 1D line-out
        print_rows(rows)
        print("\n[self-test: ZFP only. SZ3/MGARD/ROUND_FIRST run via libpressio on Colab.]")
        print("mxE/b<=1 => within bound; mxEc/b<=1 => bound preserved after correction.")
