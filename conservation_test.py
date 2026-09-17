"""
The comparison VFC should actually win: downstream conservation error.

ZFP/SZ bound pointwise error but do not re-close a global physical
invariant. This measures, for VFC vs ZFP at MATCHED pointwise error, how
far each one's reconstruction drifts on integrated quantities that
downstream science checks: total energy, total momentum (per component),
and mean (mass proxy).

If VFC has a reason to exist, it shows up here: near-zero conservation
drift where ZFP has nonzero drift at the same fidelity and similar-or-worse
size.
"""
from __future__ import annotations
import numpy as np

from attention import vorticity_attention_map
from bitpack import quantize_tiered, DEFAULT_TIERS
from constraint import correct_energy_conservation
from pipeline import make_synthetic_turbulence_field

try:
    import zfpy
    HAVE_ZFP = True
except Exception:
    HAVE_ZFP = False


def invariants(u, v, w):
    return {
        "energy": float(np.sum(u.astype(np.float64) ** 2
                               + v.astype(np.float64) ** 2
                               + w.astype(np.float64) ** 2)),
        "momentum_u": float(np.sum(u.astype(np.float64))),
        "momentum_v": float(np.sum(v.astype(np.float64))),
        "momentum_w": float(np.sum(w.astype(np.float64))),
    }


def rel_drift(base, other, near_zero=1.0):
    """Relative drift, but flagged None where the baseline is ~0 (relative
    error is meaningless on a near-zero integrated quantity, e.g. net
    momentum of a symmetric field)."""
    out = {}
    for k in base:
        scale = abs(base[k])
        if scale < near_zero:
            out[k] = None  # near-zero baseline: relative drift not meaningful
        else:
            out[k] = abs(other[k] - base[k]) / scale
    return out


def vfc_reconstruct(u, v, w, att):
    outs = []
    for f in (u, v, w):
        qf = quantize_tiered(f, att, DEFAULT_TIERS)
        _, corrected = correct_energy_conservation(f, qf)
        outs.append(corrected)
    return outs


def zfp_reconstruct(u, v, w, tol):
    return [zfpy.decompress_numpy(zfpy.compress_numpy(f.astype(np.float32), tolerance=tol))
            for f in (u, v, w)]


def match_tol(f, target_max_abs, lo=1e-6, hi=1.0, iters=14):
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        d = zfpy.decompress_numpy(zfpy.compress_numpy(f.astype(np.float32), tolerance=mid))
        if np.abs(f - d).max() > target_max_abs:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi)


def run(n=64, seed=0):
    u, v, w = make_synthetic_turbulence_field(n=n, seed=seed)
    att = vorticity_attention_map(u, v, w, smooth_sigma=0.6)
    base = invariants(u, v, w)

    ur, vr, wr = vfc_reconstruct(u, v, w, att)
    vfc_inv = invariants(ur, vr, wr)
    vfc_maxabs = max(np.abs(u - ur).max(), np.abs(v - vr).max(), np.abs(w - wr).max())
    vfc_drift = rel_drift(base, vfc_inv)

    result = {"n": n, "seed": seed, "vfc_maxabs": float(vfc_maxabs),
              "vfc_drift": vfc_drift}

    if HAVE_ZFP:
        tols = [match_tol(f, vfc_maxabs) for f in (u, v, w)]
        uz = zfpy.decompress_numpy(zfpy.compress_numpy(u.astype(np.float32), tolerance=tols[0]))
        vz = zfpy.decompress_numpy(zfpy.compress_numpy(v.astype(np.float32), tolerance=tols[1]))
        wz = zfpy.decompress_numpy(zfpy.compress_numpy(w.astype(np.float32), tolerance=tols[2]))
        zfp_inv = invariants(uz, vz, wz)
        result["zfp_drift"] = rel_drift(base, zfp_inv)

    return result


def print_run(r):
    print(f"\n[Conservation drift @ matched pointwise error]  n={r['n']} seed={r['seed']}")
    print(f"  VFC max abs error: {r['vfc_maxabs']:.5f}")
    print(f"  {'invariant':<14}{'VFC rel drift':>16}{'ZFP rel drift':>16}")
    for k in r["vfc_drift"]:
        vd = r["vfc_drift"][k]
        zd = r.get("zfp_drift", {}).get(k, None)
        vs = "   n/a (~0 base)" if vd is None else f"{vd:>16.2e}"
        zs = "   n/a (~0 base)" if zd is None else f"{zd:>16.2e}"
        print(f"  {k:<14}{vs}{zs}")


if __name__ == "__main__":
    for s in (0, 1):
        print_run(run(n=48, seed=s))
