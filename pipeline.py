"""
End-to-end orchestration: attention map -> tiered quantization -> global
constraint correction -> entropy-coded backend, plus a synthetic 3D
turbulence-like test field (Taylor-Green vortex + multi-scale noise) used
to validate and report on the whole pipeline.
"""
from __future__ import annotations
import time
import numpy as np

from attention import vorticity_attention_map
from bitpack import quantize_tiered, dequantize_tiered, tier_report, DEFAULT_TIERS
from constraint import correct_energy_conservation, sum_of_squares
from backend import serialize_quantized_field, evaluate_compression


def make_synthetic_turbulence_field(n: int = 64, seed: int = 0):
    """
    Taylor-Green-vortex base flow (smooth, large-scale structure) plus
    layered multi-octave noise (small-scale turbulence) so the field has
    both a laminar background and sharp, chaotic high-vorticity regions --
    exactly the regime this framework targets.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    u = np.cos(X) * np.sin(Y) * np.cos(Z)
    v = -np.sin(X) * np.cos(Y) * np.cos(Z)
    w = np.zeros_like(u)

    # Multi-octave noise to create localized turbulent structure.
    for octave in range(1, 5):
        scale = 1.0 / octave
        phase = rng.uniform(0, 2 * np.pi, size=3)
        u += scale * 0.3 * np.sin(octave * X + phase[0]) * np.cos(octave * Y)
        v += scale * 0.3 * np.cos(octave * Y + phase[1]) * np.sin(octave * Z)
        w += scale * 0.3 * np.sin(octave * Z + phase[2]) * np.cos(octave * X)

    # Sprinkle a few sharp localized "shock-like" bursts.
    for _ in range(6):
        cx, cy, cz = rng.integers(0, n, size=3)
        r = n // 12
        xx, yy, zz = np.ogrid[:n, :n, :n]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        burst = np.exp(-dist2 / (2 * r ** 2)) * rng.uniform(1.5, 3.0)
        u += burst
        v += burst * rng.uniform(-1, 1)

    return u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)


def run_pipeline(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                  smooth_sigma: float = 0.6, zstd_level: int = 19,
                  verbose: bool = True) -> dict:
    t0 = time.time()

    # 1. Front-end gradient pass -> attention map (vorticity magnitude)
    attention_map = vorticity_attention_map(u, v, w, smooth_sigma=smooth_sigma)
    t1 = time.time()

    # 2. Adaptive tiered quantization, routed by attention
    qf_u = quantize_tiered(u, attention_map, DEFAULT_TIERS)
    qf_v = quantize_tiered(v, attention_map, DEFAULT_TIERS)
    qf_w = quantize_tiered(w, attention_map, DEFAULT_TIERS)
    t2 = time.time()

    # 3. Global constraint correction (conserve total kinetic energy,
    #    proportional to sum of squares across all three components)
    def combined_energy(_ignored):
        return sum_of_squares(u) + sum_of_squares(v) + sum_of_squares(w)

    # Correct each component's background against a metric that reflects
    # its own share of total energy (component-wise sum of squares is a
    # valid conserved sub-quantity here since u, v, w are independent
    # arrays in this representation).
    report_u, corrected_u = correct_energy_conservation(u, qf_u)
    report_v, corrected_v = correct_energy_conservation(v, qf_v)
    report_w, corrected_w = correct_energy_conservation(w, qf_w)
    t3 = time.time()

    # 4. Backend entropy coding + evaluation against baselines
    eval_u = evaluate_compression(u, qf_u, zstd_level=zstd_level)
    eval_v = evaluate_compression(v, qf_v, zstd_level=zstd_level)
    eval_w = evaluate_compression(w, qf_w, zstd_level=zstd_level)
    t4 = time.time()

    # Error metrics (post constraint-correction) vs original
    def err_stats(orig, recon):
        diff = np.abs(orig.astype(np.float64) - recon.astype(np.float64))
        denom = np.abs(orig).max() + 1e-12
        return {
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "max_rel_error_pct": float(100 * diff.max() / denom),
        }

    err_u = err_stats(u, corrected_u)
    err_v = err_stats(v, corrected_v)
    err_w = err_stats(w, corrected_w)

    total_raw = eval_u["raw_bytes"] + eval_v["raw_bytes"] + eval_w["raw_bytes"]
    total_vfc = eval_u["vfc_stream_plus_zstd_bytes"] + eval_v["vfc_stream_plus_zstd_bytes"] + eval_w["vfc_stream_plus_zstd_bytes"]
    total_base_zstd = eval_u["baseline_zstd_on_raw_bytes"] + eval_v["baseline_zstd_on_raw_bytes"] + eval_w["baseline_zstd_on_raw_bytes"]
    total_base_gzip = eval_u["baseline_gzip_on_raw_bytes"] + eval_v["baseline_gzip_on_raw_bytes"] + eval_w["baseline_gzip_on_raw_bytes"]

    result = {
        "timing_sec": {
            "attention_map": t1 - t0,
            "quantization": t2 - t1,
            "constraint_correction": t3 - t2,
            "entropy_coding_eval": t4 - t3,
            "total": t4 - t0,
        },
        "tier_report_u": tier_report(qf_u),
        "constraint_reports": {"u": report_u, "v": report_v, "w": report_w},
        "error_stats": {"u": err_u, "v": err_v, "w": err_w},
        "compression": {
            "total_raw_bytes": total_raw,
            "total_vfc_bytes": total_vfc,
            "total_baseline_zstd_bytes": total_base_zstd,
            "total_baseline_gzip_bytes": total_base_gzip,
            "overall_ratio_vfc_vs_raw": total_raw / total_vfc,
            "overall_ratio_baseline_zstd_vs_raw": total_raw / total_base_zstd,
            "overall_ratio_baseline_gzip_vs_raw": total_raw / total_base_gzip,
            "vfc_improvement_vs_zstd_baseline": total_base_zstd / total_vfc,
        },
        "attention_map_stats": {
            "mean": float(attention_map.mean()),
            "frac_high_attention_gt_0.75": float(np.mean(attention_map >= 0.75)),
            "frac_low_attention_lt_0.15": float(np.mean(attention_map < 0.15)),
        },
    }

    if verbose:
        print_report(result)

    return result


def print_report(result: dict):
    print("=" * 78)
    print("VIEWER-FIRST COMPRESSION -- PROTOTYPE RESULTS")
    print("=" * 78)

    print("\n[Attention map]")
    stats = result["attention_map_stats"]
    print(f"  mean attention                 : {stats['mean']:.4f}")
    print(f"  fraction high-attention (>=0.75): {stats['frac_high_attention_gt_0.75']*100:.2f}%")
    print(f"  fraction low-attention  (<0.15) : {stats['frac_low_attention_lt_0.15']*100:.2f}%")

    print("\n[Tier allocation (u-component, representative)]")
    print("  " + result["tier_report_u"].replace("\n", "\n  "))

    print("\n[Global constraint correction -- energy conservation]")
    for comp, rep in result["constraint_reports"].items():
        print(f"  component {comp}: alpha={rep.alpha:.6f}  "
              f"rel_error before={rep.relative_error_before:.3e}  "
              f"after={rep.relative_error_after:.3e}")

    print("\n[Reconstruction error vs. original, post-correction]")
    for comp, e in result["error_stats"].items():
        print(f"  component {comp}: max_abs={e['max_abs_error']:.5f}  "
              f"mean_abs={e['mean_abs_error']:.5f}  rmse={e['rmse']:.5f}  "
              f"max_rel={e['max_rel_error_pct']:.3f}%")

    print("\n[Compression ratios, combined u+v+w]")
    c = result["compression"]
    print(f"  raw float32 size                : {c['total_raw_bytes']:>10} bytes")
    print(f"  VFC tiered stream + zstd         : {c['total_vfc_bytes']:>10} bytes "
          f"({c['overall_ratio_vfc_vs_raw']:.2f}x vs raw)")
    print(f"  baseline: zstd on raw floats     : {c['total_baseline_zstd_bytes']:>10} bytes "
          f"({c['overall_ratio_baseline_zstd_vs_raw']:.2f}x vs raw)")
    print(f"  baseline: gzip on raw floats     : {c['total_baseline_gzip_bytes']:>10} bytes "
          f"({c['overall_ratio_baseline_gzip_vs_raw']:.2f}x vs raw)")
    print(f"  VFC improvement over zstd baseline: {c['vfc_improvement_vs_zstd_baseline']:.2f}x")

    print("\n[Timing]")
    for k, v in result["timing_sec"].items():
        print(f"  {k:<24}: {v*1000:8.2f} ms")

    print("=" * 78)


if __name__ == "__main__":
    u, v, w = make_synthetic_turbulence_field(n=64, seed=42)
    run_pipeline(u, v, w)
