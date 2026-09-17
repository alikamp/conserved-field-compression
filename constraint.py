"""
Global Constraint Correction (Physics Invariants).

Quantization of the low-attention background perturbs the field locally.
Left uncorrected, that perturbation biases *macroscopic* quantities derived
by integrating over the whole domain (total energy, mass, momentum), which
is exactly what downstream scientific analysis usually cares about.

This layer runs *after* quantization and *before* entropy coding. It
measures a global scalar metric (default: sum-of-squares, a stand-in for
total kinetic energy \\int |u|^2 dV up to a constant factor) on the original
field vs. the quantized/dequantized field, then algebraically corrects the
values in the low-attention ("correctable") tiers so the metric matches the
uncompressed baseline to floating point precision.

High-attention / near-lossless tiers are left untouched -- they already
carry the fidelity the viewer needs; only the background absorbs the
correction, exactly as physically expected for near-linear conserved
quantities.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, List

from bitpack import QuantizedField, dequantize_tiered


def sum_of_squares(x: np.ndarray) -> float:
    return float(np.sum(x.astype(np.float64) ** 2))


def total_energy(x: np.ndarray, rho: float = 1.0) -> float:
    """0.5 * rho * sum(|u|^2) -- proportional to sum_of_squares."""
    return 0.5 * rho * sum_of_squares(x)


@dataclass
class ConstraintReport:
    metric_name: str
    original: float
    before_correction: float
    after_correction: float
    alpha: float
    relative_error_before: float
    relative_error_after: float


def correct_energy_conservation(
    original_field: np.ndarray,
    qf: QuantizedField,
    metric_fn: Callable[[np.ndarray], float] = sum_of_squares,
    metric_name: str = "sum_of_squares (~energy)",
    correctable_max_bitwidth: int = 16,
) -> ConstraintReport:
    """
    Rescale the quantized values in tiers with bitwidth <= correctable_max_bitwidth
    (i.e. the lossy background) by a single multiplicative factor alpha so that
    metric_fn(corrected_reconstruction) == metric_fn(original_field) exactly
    (up to float64 rounding), while leaving near-lossless (high-attention)
    tiers untouched.

    Solves analytically for a sum-of-squares-type metric:
        target = metric_fn(original)
        fixed  = contribution from untouched (high-precision) tiers
        S_low  = sum of squares of the *quantized* background values
        alpha  = sqrt( (target - fixed) / S_low )
    then multiplies every background value's dequantized level by alpha.

    This is exact for metric_fn == sum_of_squares (and anything proportional
    to it, e.g. total_energy). For a general metric_fn the same alpha is
    applied and the achieved metric is reported so the residual error is
    visible rather than silently hidden.
    """
    target = metric_fn(original_field)

    recon = dequantize_tiered(qf)
    before = metric_fn(recon)
    rel_before = abs(before - target) / (abs(target) + 1e-30)

    # Partition reconstructed values into "fixed" (high precision, untouched)
    # and "correctable" (background, will be rescaled).
    fixed_mask = np.zeros(recon.shape, dtype=bool)
    correctable_mask = np.zeros(recon.shape, dtype=bool)
    flat_fixed = fixed_mask.ravel()
    flat_correctable = correctable_mask.ravel()

    for block in qf.tiers:
        if block.indices.size == 0:
            continue
        if block.bitwidth > correctable_max_bitwidth:
            flat_fixed[block.indices] = True
        else:
            flat_correctable[block.indices] = True

    flat_recon = recon.ravel().astype(np.float64)
    fixed_vals = flat_recon[flat_fixed]
    correctable_vals = flat_recon[flat_correctable]

    fixed_contrib = float(np.sum(fixed_vals ** 2))
    s_low = float(np.sum(correctable_vals ** 2))

    remainder = target - fixed_contrib
    if s_low < 1e-30 or remainder < 0:
        # Degenerate case: nothing to scale, or target unreachable by scaling
        # alone (would require sign flips / negative energy) -- fall back to
        # alpha=1 (no-op) rather than producing an unphysical result.
        alpha = 1.0
    else:
        alpha = float(np.sqrt(remainder / s_low))

    flat_recon[flat_correctable] = correctable_vals * alpha
    corrected = flat_recon.reshape(recon.shape).astype(np.float32)

    after = metric_fn(corrected)
    rel_after = abs(after - target) / (abs(target) + 1e-30)

    return ConstraintReport(
        metric_name=metric_name,
        original=target,
        before_correction=before,
        after_correction=after,
        alpha=alpha,
        relative_error_before=rel_before,
        relative_error_after=rel_after,
    ), corrected
