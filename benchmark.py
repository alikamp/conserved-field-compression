"""
Extended benchmark harness for the Viewer-First Compression prototype.

Produces the concrete data points needed for an open-source release:
  - scaling across grid sizes (ratio + timing)
  - variance across random seeds (mean +/- std)
  - a FAIR, matched-error comparison against ZFP (zfpy), the actual
    state-of-the-art lossy scientific compressor -- not just lossless zstd.

Fairness note: VFC is lossy. Comparing it to lossless zstd flatters it.
The honest comparison tunes ZFP's tolerance so ZFP's max abs error is
close to VFC's, then compares compression ratios at that matched fidelity.
"""
from __future__ import annotations
import time
import numpy as np

from attention import vorticity_attention_map
from bitpack import quantize_tiered, dequantize_tiered, DEFAULT_TIERS
from constraint import correct_energy_conservation, sum_of_squares
from backend import serialize_quantized_field, compress_zstd, compress_gzip
from pipeline import make_synthetic_turbulence_field

try:
    import zfpy
    HAVE_ZFP = True
except Exception:
    HAVE_ZFP = False


def _err_stats(orig, recon):
    diff = np.abs(orig.astype(np.float64) - recon.astype(np.float64))
    denom = np.abs(orig).max() + 1e-12
    return {
        "max_abs": float(diff.max()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_rel_pct": float(100 * diff.max() / denom),
    }


def vfc_compress_component(field, attention_map, zstd_level=19):
    """Full VFC on one component; returns (compressed_bytes, reconstruction, err)."""
    qf = quantize_tiered(field, attention_map, DEFAULT_TIERS)
    report, corrected = correct_energy_conservation(field, qf)
    stream = serialize_quantized_field(qf)
    comp = compress_zstd(stream, level=zstd_level)
    return len(comp), corrected, _err_stats(field, corrected), report


def zfp_compress_component(field, tolerance):
    comp = zfpy.compress_numpy(field.astype(np.float32), tolerance=tolerance)
    recon = zfpy.decompress_numpy(comp)
    return len(comp), recon, _err_stats(field, recon)


def match_zfp_tolerance(field, target_max_abs, lo=1e-6, hi=1.0, iters=18):
    """Bisect ZFP tolerance so its max abs error ~ target_max_abs."""
    if not HAVE_ZFP:
        return None
    best = hi
    for _ in range(iters):
        mid = np.sqrt(lo * hi)  # geometric bisection
        _, _, err = zfp_compress_component(field, mid)
        if err["max_abs"] > target_max_abs:
            hi = mid
        else:
            best = mid
            lo = mid
    return best


def run_one(n, seed, zstd_level=19, smooth_sigma=0.6):
    u, v, w = make_synthetic_turbulence_field(n=n, seed=seed)
    comps = {"u": u, "v": v, "w": w}

    t0 = time.time()
    att = vorticity_attention_map(u, v, w, smooth_sigma=smooth_sigma)
    t_att = time.time() - t0

    raw_total = 0
    vfc_total = 0
    zstd_total = 0
    gzip_total = 0
    zfp_total = 0
    max_abs_vfc = 0.0
    rmse_accum = []

    t0 = time.time()
    for name, f in comps.items():
        raw = f.astype(np.float32).tobytes()
        raw_total += len(raw)

        c_vfc, recon, err, _rep = vfc_compress_component(f, att, zstd_level)
        vfc_total += c_vfc
        max_abs_vfc = max(max_abs_vfc, err["max_abs"])
        rmse_accum.append(err["rmse"])

        zstd_total += len(compress_zstd(raw, level=zstd_level))
        gzip_total += len(compress_gzip(raw))
    t_vfc = time.time() - t0

    row = {
        "n": n,
        "seed": seed,
        "voxels": n ** 3,
        "raw_bytes": raw_total,
        "vfc_bytes": vfc_total,
        "zstd_bytes": zstd_total,
        "gzip_bytes": gzip_total,
        "vfc_ratio": raw_total / vfc_total,
        "zstd_ratio": raw_total / zstd_total,
        "gzip_ratio": raw_total / gzip_total,
        "vfc_max_abs": max_abs_vfc,
        "vfc_rmse": float(np.mean(rmse_accum)),
        "t_attention_s": t_att,
        "t_vfc_s": t_vfc,
    }

    # Matched-error ZFP comparison (fair, lossy-vs-lossy)
    if HAVE_ZFP:
        zfp_err_accum = []
        for name, f in comps.items():
            tol = match_zfp_tolerance(f, max_abs_vfc)
            c_zfp, recon_zfp, err_zfp = zfp_compress_component(f, tol)
            zfp_total += c_zfp
            zfp_err_accum.append(err_zfp["max_abs"])
        row["zfp_bytes"] = zfp_total
        row["zfp_ratio"] = raw_total / zfp_total
        row["zfp_max_abs_matched"] = float(np.mean(zfp_err_accum))
        row["vfc_vs_zfp"] = zfp_total / vfc_total  # >1 means VFC smaller

    return row


def run_suite(grid_sizes=(32, 48, 64), seeds=(0, 1, 2), zstd_level=19, verbose=True):
    rows = []
    for n in grid_sizes:
        for s in seeds:
            row = run_one(n, s, zstd_level=zstd_level)
            rows.append(row)
            if verbose:
                extra = ""
                if "zfp_ratio" in row:
                    extra = (f"  zfp={row['zfp_ratio']:.2f}x"
                             f"  vfc/zfp={row['vfc_vs_zfp']:.2f}x")
                print(f"n={n:>3} seed={s}  vfc={row['vfc_ratio']:.2f}x  "
                      f"zstd={row['zstd_ratio']:.2f}x{extra}  "
                      f"vfc_maxabs={row['vfc_max_abs']:.4f}  "
                      f"t={row['t_vfc_s']*1000:.0f}ms")
    return rows


def summarize(rows):
    import statistics as st
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r)
    print("\n" + "=" * 78)
    print("SUMMARY (mean over seeds)")
    print("=" * 78)
    header = f"{'grid':>6} {'vfc_ratio':>10} {'zstd_ratio':>10}"
    if HAVE_ZFP:
        header += f" {'zfp_ratio':>10} {'vfc/zfp':>8}"
    header += f" {'vfc_maxabs':>11} {'t_vfc_ms':>9}"
    print(header)
    for n, rs in sorted(by_n.items()):
        vfc = st.mean(r["vfc_ratio"] for r in rs)
        zstd = st.mean(r["zstd_ratio"] for r in rs)
        maxabs = st.mean(r["vfc_max_abs"] for r in rs)
        t = st.mean(r["t_vfc_s"] for r in rs) * 1000
        line = f"{n:>6} {vfc:>10.3f} {zstd:>10.3f}"
        if HAVE_ZFP:
            zfp = st.mean(r["zfp_ratio"] for r in rs)
            vz = st.mean(r["vfc_vs_zfp"] for r in rs)
            line += f" {zfp:>10.3f} {vz:>8.3f}"
        line += f" {maxabs:>11.5f} {t:>9.1f}"
        print(line)


if __name__ == "__main__":
    rows = run_suite(grid_sizes=(32, 48), seeds=(0, 1))
    summarize(rows)
