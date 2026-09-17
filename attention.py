"""
Front-end gradient pass: builds a normalized [0,1] "attention map" over a
scientific float array, highlighting structurally important regions
(shock fronts, vortices, sharp gradients) vs. smooth/laminar background.

Supports 2D scalar fields, 2D vector fields (u, v), and 3D vector fields
(u, v, w) via curl/vorticity magnitude. Falls back to gradient-magnitude
for scalar fields.
"""
from __future__ import annotations
import numpy as np


def _minmax_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def gradient_magnitude_map(field: np.ndarray) -> np.ndarray:
    """Scalar field -> normalized gradient-magnitude attention map."""
    grads = np.gradient(field.astype(np.float64))
    if field.ndim == 1:
        grads = [grads]
    mag = np.zeros_like(field, dtype=np.float64)
    for g in grads:
        mag += g ** 2
    mag = np.sqrt(mag)
    return _minmax_normalize(mag)


def curl_2d(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """2D scalar vorticity: omega = dv/dx - du/dy."""
    dv_dx = np.gradient(v, axis=1)
    du_dy = np.gradient(u, axis=0)
    return dv_dx - du_dy


def curl_3d(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """3D vorticity vector omega = grad x (u,v,w), returns magnitude field."""
    du_dy = np.gradient(u, axis=1)
    du_dz = np.gradient(u, axis=2)
    dv_dx = np.gradient(v, axis=0)
    dv_dz = np.gradient(v, axis=2)
    dw_dx = np.gradient(w, axis=0)
    dw_dy = np.gradient(w, axis=1)

    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy

    mag = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)
    return mag


def vorticity_attention_map(*components: np.ndarray, smooth_sigma: float = 0.0) -> np.ndarray:
    """
    Build a normalized [0,1] attention map from a velocity field.

    - 1 component  -> scalar field, uses gradient magnitude.
    - 2 components -> 2D vector field (u, v), uses curl (vorticity).
    - 3 components -> 3D vector field (u, v, w), uses curl magnitude.

    smooth_sigma: optional Gaussian smoothing (via scipy) applied to the
    raw structural signal before normalization, to reduce single-voxel
    noise dominating attention allocation.
    """
    if len(components) == 1:
        raw = np.abs(np.gradient(components[0].astype(np.float64))[0]) \
            if components[0].ndim == 1 else None
        att_map = gradient_magnitude_map(components[0])
        raw_signal = None
    elif len(components) == 2:
        u, v = components
        raw_signal = np.abs(curl_2d(u.astype(np.float64), v.astype(np.float64)))
        att_map = None
    elif len(components) == 3:
        u, v, w = components
        raw_signal = curl_3d(u.astype(np.float64), v.astype(np.float64), w.astype(np.float64))
        att_map = None
    else:
        raise ValueError("Expected 1 (scalar), 2 (2D vector) or 3 (3D vector) components")

    if att_map is not None:
        return att_map

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        raw_signal = gaussian_filter(raw_signal, sigma=smooth_sigma)

    return _minmax_normalize(raw_signal)
