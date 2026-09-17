"""
Head-to-head: original tiered VFC vs blocked VFC (#3) vs blocked+downsample
(#4) vs ZFP, all at matched-error-tuned ZFP for a fair lossy comparison.

For each method we report: compression ratio (after zstd), max abs error,
and energy conservation drift. ZFP is tuned per method to that method's own
max error, so the ratio comparison is apples-to-apples on fidelity.
"""
from __future__ import annotations
import numpy as np

from attention import vorticity_attention_map
from bitpack import quantize_tiered, dequantize_tiered, DEFAULT_TIERS
from constraint import correct_energy_conservation
from backend import serialize_quantized_field, compress_zstd
from blocked import quantize_blocked, dequantize_blocked
from pipeline import make_synthetic_turbulence_field

try:
    import zfpy
    HAVE_ZFP = True
except Exception:
    HAVE_ZFP = False


def _err(orig, rec):
    d = np.abs(orig.astype(np.float64) - rec.astype(np.float64))
    return float(d.max())


def _energy_drift(orig, rec):
    t = float(np.sum(orig.astype(np.float64) ** 2))
    r = float(np.sum(rec.astype(np.float64) ** 2))
    return abs(r - t) / (abs(t) + 1e-30)


def tiered(field, att, level=19):
    qf = quantize_tiered(field, att, DEFAULT_TIERS)
    _, corrected = correct_energy_conservation(field, qf)
    comp = compress_zstd(serialize_quantized_field(qf), level=level)
    return len(comp), corrected


def blocked(field, att, B, ds, level=19):
    bf = quantize_blocked(field, att, block_size=B, downsample_low=ds)
    rec = dequantize_blocked(bf)
    comp = compress_zstd(bf.stream, level=level)
    return len(comp), rec


def zfp_matched(field, target_max_abs, lo=1e-7, hi=2.0, iters=22):
    lo0, hi0 = lo, hi
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        d = zfpy.decompress_numpy(zfpy.compress_numpy(field.astype(np.float32), tolerance=mid))
        if np.abs(field - d).max() > target_max_abs:
            hi = mid
        else:
            lo = mid
    tol = np.sqrt(lo * hi)
    comp = zfpy.compress_numpy(field.astype(np.float32), tolerance=tol)
    rec = zfpy.decompress_numpy(comp)
    return len(comp), rec


def run(n=64, seed=0, level=19):
    u, v, w = make_synthetic_turbulence_field(n=n, seed=seed)
    att = vorticity_attention_map(u, v, w, smooth_sigma=0.6)
    comps = {"u": u, "v": v, "w": w}
    raw = sum(f.nbytes for f in comps.values())

    methods = {
        "tiered (orig)":    lambda f: tiered(f, att, level),
        "blocked B8":       lambda f: blocked(f, att, 8, False, level),
        "blocked B4":       lambda f: blocked(f, att, 4, False, level),
        "blocked B4 +ds":   lambda f: blocked(f, att, 4, True, level),
    }

    print(f"\n{'='*86}\nn={n} seed={seed}   raw={raw} bytes\n{'='*86}")
    header = f"{'method':<18}{'ratio':>9}{'max_err':>11}{'E-drift':>11}"
    if HAVE_ZFP:
        header += f"{'ZFP ratio':>11}{'VFC/ZFP':>9}"
    print(header)

    results = {}
    for name, fn in methods.items():
        tot = 0; maxerr = 0.0; recs = {}
        for cn, f in comps.items():
            nbytes, rec = fn(f)
            tot += nbytes
            maxerr = max(maxerr, _err(f, rec))
            recs[cn] = rec
        ratio = raw / tot
        edrift = _energy_drift(
            np.stack(list(comps.values())), np.stack([recs[c] for c in comps]))
        line = f"{name:<18}{ratio:>9.2f}{maxerr:>11.4f}{edrift:>11.1e}"

        if HAVE_ZFP:
            ztot = 0
            for cn, f in comps.items():
                znbytes, _ = zfp_matched(f, maxerr)
                ztot += znbytes
            zratio = raw / ztot
            line += f"{zratio:>11.2f}{ratio/zratio:>9.2f}"
        print(line)
        results[name] = {"ratio": ratio, "max_err": maxerr, "e_drift": edrift}

    return results


if __name__ == "__main__":
    for n in (32, 64):
        run(n=n, seed=0)
