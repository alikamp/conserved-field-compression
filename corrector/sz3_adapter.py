"""
sz3_adapter.py -- Alika Parks
SZ3 for the rd_full.py harness without libpressio/conda: SZ3 runs as an HDF5
filter from the pip package hdf5plugin (absolute-error mode).

  pip install h5py hdf5plugin zfpy

Provides
  SZ3            codec adapter with the same interface as rd_full.ZFP:
                 compress(field, abs_tol) -> (compressed_bytes, reconstruction)
  study_tight    paper runs (Table "Across codecs"): compress at (1-eps)*b with
                 b = NOA bound * value range, apply mass+energy correction, report
                 ratio, max-error/bound, drift layer off/on; if the corrected error
                 exceeds b, retighten and recompress (bound-safe).
  cr_impact      paper runs (Table "Cost in compression ratio"): ratio at the full
                 bound without the layer vs. at 0.9*b with it (+16 bytes).

Usage
  python sz3_adapter.py                  # self-test on a synthetic 3D field
  python sz3_adapter.py field.npy [name] # real field (any dim, any float dtype)
"""
import io, sys, numpy as np
from rd_full import invariants, conserve, ZFP, synth3d

FRACS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)          # NOA bounds (fraction of value range)

class SZ3:
    name = "sz3"
    def compress(self, field, abs_tol):
        import h5py, hdf5plugin
        f = np.ascontiguousarray(field, dtype=np.float32); buf = io.BytesIO()
        with h5py.File(buf, "w") as h:
            d = h.create_dataset("x", data=f, chunks=f.shape,
                                 **hdf5plugin.SZ3(absolute=float(abs_tol)))
            nbytes = d.id.get_storage_size()
        buf.seek(0)
        with h5py.File(buf, "r") as h: rec = h["x"][...]
        return int(nbytes), rec

def _one(field, codec, b_used, M0, S0):
    nbytes, rec = codec.compress(field, b_used)
    fc = conserve(rec, M0, S0)
    me = float(np.abs(field.astype(np.float64) - fc.astype(np.float64)).max())
    return nbytes, rec, fc, me

def study_tight(field, name, codec, fracs=FRACS, eps=0.10):
    field = np.ascontiguousarray(field.astype(np.float32))
    rng = float(field.max() - field.min()); M0, S0 = invariants(field); rows = []
    Mabs = float(np.abs(field.astype(np.float64)).sum())   # mass reference for signed fields
    rel = lambda x, x0, ref: abs(x - x0) / ref
    for frac in fracs:
        b = frac * rng; bu = (1 - eps) * b
        nb, rec, fc, me = _one(field, codec, bu, M0, S0); k = 0
        while me > b and k < 4:                  # bound-safe retighten (rarely needed)
            bu *= 0.98 * b / me; k += 1
            nb, rec, fc, me = _one(field, codec, bu, M0, S0)
        Mh, Sh = invariants(rec); Mc, Sc = invariants(fc)
        rows.append(dict(codec=codec.name, field=name, frac=frac, ratio=field.nbytes / nb,
                         mxc=me / b, retight=k,
                         E_raw=rel(Sh, S0, S0), E_cor=rel(Sc, S0, S0),
                         M_raw=rel(Mh, M0, Mabs), M_cor=rel(Mc, M0, Mabs)))
    return rows

def print_tight(rows):
    h = (f"{'codec':<5} {'field':<16}{'NOA eps':>8}{'ratio':>9}{'maxerr/b':>9}{'rt':>3}"
         f"{'E raw':>10}{'E +layer':>10}{'M raw':>10}{'M +layer':>10}")
    print(h); print('-' * len(h))
    for r in rows:
        print(f"{r['codec']:<5} {r['field']:<16}{r['frac']:>8.0e}{r['ratio']:>9.1f}{r['mxc']:>9.2f}"
              f"{r['retight']:>3}{r['E_raw']:>10.2e}{r['E_cor']:>10.2e}{r['M_raw']:>10.2e}{r['M_cor']:>10.2e}")
    print("maxerr/b <= 1 is the pointwise guarantee; rt = retightens; M drift relative to sum|x|.")

def cr_impact(field, name, codec, fracs=FRACS, headroom=0.9, scalar_bytes=16):
    f = np.ascontiguousarray(field.astype(np.float32)); R = float(f.max() - f.min())
    print(f"\n{codec.name}  {name}:  NOA eps | ratio, no layer (eps) | ratio, layer "
          f"({headroom} eps + {scalar_bytes} B) | cost")
    for frac in fracs:
        b = frac * R
        n_full, _ = codec.compress(f, b)
        n_tight, _ = codec.compress(f, headroom * b)
        r0, r1 = f.nbytes / n_full, f.nbytes / (n_tight + scalar_bytes)
        print(f"   {frac:.0e}   {r0:10.1f}   {r1:10.1f}   {1 - r1/r0:6.1%}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        F = np.load(sys.argv[1]); name = sys.argv[2] if len(sys.argv) > 2 else "field"
    else:
        F = synth3d(); name = "synth3d"
    rows = []
    for codec in (ZFP(), SZ3()): rows += study_tight(F, name, codec)
    print_tight(rows)
    for codec in (ZFP(), SZ3()): cr_impact(F, name, codec)
